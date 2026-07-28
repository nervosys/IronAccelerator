//! Opt-out OpenTelemetry export.
//!
//! Enabled by default; remove it with `default-features = false` (or by
//! disabling the `telemetry` feature) — that is the opt-out. When enabled,
//! [`init_from_env`] is called once from [`Runtime::new`](crate::Runtime::new).
//!
//! # What it does and, deliberately, does not do
//!
//! This crate ships **no endpoint and no credential**. It is destination
//! neutral: it exports only where the *operator* points it, using the standard
//! OpenTelemetry environment variables read at runtime —
//! `OTEL_EXPORTER_OTLP_ENDPOINT`, `OTEL_EXPORTER_OTLP_HEADERS`,
//! `OTEL_EXPORTER_OTLP_PROTOCOL`, `OTEL_SERVICE_NAME`. If
//! `OTEL_EXPORTER_OTLP_ENDPOINT` is unset, nothing is exported and no network
//! connection is made.
//!
//! That is the whole safety story:
//!
//! - **Nothing happens at build or install time.** There is no build script.
//!   Export is a runtime concern of the running process, not of anyone who
//!   compiles the crate.
//! - **No secret lives in the published source.** A token baked into a crate on
//!   crates.io is public and permanent; this reads the operator's own token
//!   from the operator's own environment instead.
//! - **A deployment that wants telemetry sets these vars in its own
//!   environment** — which is exactly how a first-party service collects from
//!   its own deployments, and is the only party entitled to that data.
//!
//! Set `IRONACCEL_TELEMETRY=off` to suppress even that, e.g. for a process that
//! configures its own OpenTelemetry pipeline and does not want this crate
//! installing a second one.

use std::sync::atomic::{AtomicBool, Ordering};

static INITIALISED: AtomicBool = AtomicBool::new(false);

/// Whether the operator has asked this crate not to touch telemetry.
fn disabled_by_operator() -> bool {
    matches!(
        std::env::var("IRONACCEL_TELEMETRY").ok().as_deref(),
        Some("off") | Some("0") | Some("false") | Some("disabled")
    )
}

/// Initialise OTLP export from the ambient environment, at most once.
///
/// Returns `true` if an exporter was installed, `false` if telemetry was
/// disabled, already initialised, or no endpoint was configured. Never panics
/// and never blocks on the network; a misconfigured endpoint surfaces later on
/// the exporter's own background path, not here.
pub fn init_from_env() -> bool {
    if disabled_by_operator() {
        return false;
    }
    // No destination configured → do nothing at all. This is what keeps a
    // build with the feature on, but no operator config, completely silent.
    if std::env::var_os("OTEL_EXPORTER_OTLP_ENDPOINT").is_none() {
        return false;
    }
    if INITIALISED.swap(true, Ordering::AcqRel) {
        return false;
    }
    imp::install()
}

#[cfg(feature = "telemetry")]
mod imp {
    use opentelemetry::trace::TracerProvider as _;
    use opentelemetry::KeyValue;

    pub(super) fn install() -> bool {
        // Endpoint, headers (including any Authorization the operator set),
        // protocol, and service name are all read from the environment by the
        // OTLP builder's `with_env`-equivalent defaults. We pass none of them
        // in code, so none of them can be embedded here.
        let service =
            std::env::var("OTEL_SERVICE_NAME").unwrap_or_else(|_| "ironaccelerator".to_string());

        let exporter = match opentelemetry_otlp::SpanExporter::builder()
            .with_http()
            .build()
        {
            Ok(e) => e,
            // A bad endpoint or unreachable collector must never take down the
            // host application. Telemetry is best-effort by definition.
            Err(_) => return false,
        };

        // Simple (synchronous) exporter with the blocking HTTP client, so this
        // needs no async runtime — the driver crate does not pull in tokio.
        let provider = opentelemetry_sdk::trace::TracerProvider::builder()
            .with_simple_exporter(exporter)
            .with_resource(opentelemetry_sdk::Resource::new([KeyValue::new(
                "service.name",
                service,
            )]))
            .build();

        let _ = provider.tracer("ironaccelerator");
        opentelemetry::global::set_tracer_provider(provider);
        true
    }
}

#[cfg(not(feature = "telemetry"))]
mod imp {
    pub(super) fn install() -> bool {
        false
    }
}

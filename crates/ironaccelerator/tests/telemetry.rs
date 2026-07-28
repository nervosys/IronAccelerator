//! The opt-out telemetry must stay silent unless the operator configures it,
//! and must ship no endpoint or credential of its own.

use ironaccelerator::telemetry;

#[test]
fn does_nothing_without_an_operator_endpoint() {
    // With no OTEL_EXPORTER_OTLP_ENDPOINT in the environment, init must be a
    // no-op: no exporter, no network, no panic.
    std::env::remove_var("OTEL_EXPORTER_OTLP_ENDPOINT");
    std::env::remove_var("IRONACCEL_TELEMETRY");
    assert!(
        !telemetry::init_from_env(),
        "telemetry initialised with no endpoint configured"
    );
}

#[test]
fn operator_can_disable_even_when_configured() {
    std::env::set_var("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:4318");
    std::env::set_var("IRONACCEL_TELEMETRY", "off");
    assert!(
        !telemetry::init_from_env(),
        "IRONACCEL_TELEMETRY=off did not suppress telemetry"
    );
    std::env::remove_var("OTEL_EXPORTER_OTLP_ENDPOINT");
    std::env::remove_var("IRONACCEL_TELEMETRY");
}

/// The published source must carry no destination and no credential — those
/// come only from the operator's environment.
#[test]
fn source_embeds_no_endpoint_or_token() {
    let src = include_str!("../src/telemetry.rs");
    for needle in ["nervosys.ai", "Bearer ", "://"] {
        assert!(
            !src.contains(needle),
            "telemetry source embeds `{needle}` — endpoints and tokens must be \
             read from the environment, never compiled in"
        );
    }
}

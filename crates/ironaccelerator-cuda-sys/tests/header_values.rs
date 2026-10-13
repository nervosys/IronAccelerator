//! Pins hand-written FFI values that once disagreed with the vendor headers.
//!
//! Every value here was checked against the shipping header (CUDA 13.3
//! `cuda.h` / `library_types.h` / `cusparse.h` / `cupti_result.h`, cuDNN 9.27
//! `cudnn_graph.h`, cuTENSOR 2.8 `cutensor/types.h`). A wrong tag is not a
//! cosmetic bug: it sends the library a different request, or decodes its
//! answer as the wrong error.

use iron_cuda_sys::cublas_lt::CudaDataType;
use iron_cuda_sys::cudnn::{self, CudnnStatus};
use iron_cuda_sys::cupti::CuptiResult;
use iron_cuda_sys::cusparse::CusparseOrder;
use iron_cuda_sys::cutensor::{CutensorStatus, CutensorWorksizePref};
use iron_cuda_sys::driver::CUresult;

#[test]
fn cuda_result_codes() {
    assert_eq!(CUresult::IllegalAddress.raw(), 700);
    assert_eq!(CUresult::LaunchFailed.raw(), 719);
    assert_eq!(CUresult::from_raw(700), CUresult::IllegalAddress);
    assert_eq!(CUresult::from_raw(719), CUresult::LaunchFailed);
}

#[test]
fn narrow_float_data_types() {
    assert_eq!(CudaDataType::R8F_E4M3 as u32, 28);
    assert_eq!(CudaDataType::R8F_UE4M3 as u32, 28, "alias of R8F_E4M3");
    assert_eq!(CudaDataType::R8F_UE8M0 as u32, 30);
    assert_eq!(CudaDataType::R6F_E2M3 as u32, 31);
    assert_eq!(CudaDataType::R6F_E3M2 as u32, 32);
    assert_eq!(CudaDataType::R4F_E2M1 as u32, 33);
}

#[test]
fn sparse_order() {
    assert_eq!(CusparseOrder::Col as u32, 1);
    assert_eq!(CusparseOrder::Row as u32, 2);
}

#[test]
fn cupti_result_codes() {
    assert_eq!(CuptiResult::from_raw(13), CuptiResult::NotReady);
    assert_eq!(CuptiResult::from_raw(14), CuptiResult::NotCompatible);
    assert_eq!(CuptiResult::from_raw(15), CuptiResult::NotInitialized);
    assert_eq!(CuptiResult::from_raw(27), CuptiResult::NotSupported);
    assert_eq!(CuptiResult::from_raw(10).name(), None);
}

#[test]
fn cudnn9_status_codes() {
    assert_eq!(CudnnStatus::from_raw(0), CudnnStatus::Success);
    assert_eq!(CudnnStatus::from_raw(1001), CudnnStatus::NotInitialized);
    assert_eq!(CudnnStatus::from_raw(2000), CudnnStatus::BadParam);
    assert_eq!(CudnnStatus::from_raw(3000), CudnnStatus::NotSupported);
    assert_eq!(CudnnStatus::from_raw(3007), CudnnStatus::ArchMismatch);
    assert_eq!(CudnnStatus::from_raw(4000), CudnnStatus::InternalError);
    assert_eq!(CudnnStatus::from_raw(5000), CudnnStatus::ExecutionFailed);
    assert_eq!(CudnnStatus::from_raw(9).name(), None);
}

#[test]
fn cudnn9_backend_attribute_ids() {
    assert_eq!(cudnn::CUDNN_ATTR_EXECUTION_PLAN_HANDLE, 400);
    assert_eq!(cudnn::CUDNN_ATTR_OPERATIONGRAPH_HANDLE, 800);
    assert_eq!(cudnn::CUDNN_ATTR_OPERATIONGRAPH_OPS, 801);
    assert_eq!(cudnn::CUDNN_ATTR_TENSOR_DIMENSIONS, 902);
    assert_eq!(cudnn::CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS, 1001);
    assert_eq!(cudnn::CUDNN_ATTR_ENGINE_OPERATION_GRAPH, 1300);
    assert_eq!(cudnn::CUDNN_ATTR_OPERATION_MATMUL_ADESC, 1520);
    assert_eq!(cudnn::CUDNN_POINTWISE_SUB, 9);
    assert_eq!(cudnn::CUDNN_POINTWISE_CMP_GE, 303);
}

#[test]
fn cutensor2_values() {
    assert_eq!(CutensorWorksizePref::MinWorkspace as u32, 1);
    assert_eq!(CutensorWorksizePref::Recommended as u32, 2);
    assert_eq!(CutensorWorksizePref::Max as u32, 3);
    assert_eq!(
        CutensorStatus::from_raw(19),
        CutensorStatus::InsufficientWorkspace
    );
}

/// The point of the newtype codes: any value a library returns is
/// representable — no undefined behaviour, no information lost.
#[test]
fn unknown_codes_round_trip() {
    for raw in [1u32, 46, 222, 999, 12345, u32::MAX] {
        let r = CUresult::from_raw(raw);
        assert_eq!(r.raw(), raw);
    }
    let unknown = CUresult::from_raw(12345);
    assert_eq!(unknown.name(), None);
    assert_eq!(format!("{unknown:?}"), "CUresult(12345)");
    assert_eq!(
        format!("{:?}", CUresult::IllegalAddress),
        "CUresult::IllegalAddress"
    );
    assert_eq!(std::mem::size_of::<CUresult>(), 4);
}

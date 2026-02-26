#if canImport(Accelerate)

import Accelerate

/// Abstracts vDSP/CBLAS operations over `Float` and `Double` so
/// Accelerate-optimized `Tensor` extensions can be written once.
///
/// Only `Float` and `Double` conform. The protocol is public so that
/// overload resolution can prefer the Accelerate specializations over
/// the generic `AdditiveArithmetic`/`Numeric`/`FloatingPoint` versions.
public protocol AccelerateFloatingPoint: FloatingPoint {
    static func _vDSPSum(_ vector: UnsafeBufferPointer<Self>) -> Self
    static func _vDSPMean(_ vector: UnsafeBufferPointer<Self>) -> Self
    static func _vDSPDot(_ lhs: UnsafeBufferPointer<Self>, _ rhs: UnsafeBufferPointer<Self>) -> Self
    static func _cblasGemm(
        m: Int32, n: Int32, k: Int32,
        a: UnsafePointer<Self>, lda: Int32,
        b: UnsafePointer<Self>, ldb: Int32,
        c: UnsafeMutablePointer<Self>, ldc: Int32
    )
}

// MARK: - Float conformance

extension Float: AccelerateFloatingPoint {
    public static func _vDSPSum(_ vector: UnsafeBufferPointer<Float>) -> Float {
        vDSP.sum(vector)
    }

    public static func _vDSPMean(_ vector: UnsafeBufferPointer<Float>) -> Float {
        vDSP.mean(vector)
    }

    public static func _vDSPDot(_ lhs: UnsafeBufferPointer<Float>, _ rhs: UnsafeBufferPointer<Float>) -> Float {
        var result: Float = 0
        vDSP_dotpr(lhs.baseAddress!, 1, rhs.baseAddress!, 1, &result, vDSP_Length(lhs.count))
        return result
    }

    public static func _cblasGemm(
        m: Int32, n: Int32, k: Int32,
        a: UnsafePointer<Float>, lda: Int32,
        b: UnsafePointer<Float>, ldb: Int32,
        c: UnsafeMutablePointer<Float>, ldc: Int32
    ) {
        cblas_sgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            m, n, k,
            1.0, a, lda,
            b, ldb,
            0.0, c, ldc
        )
    }
}

// MARK: - Double conformance

extension Double: AccelerateFloatingPoint {
    public static func _vDSPSum(_ vector: UnsafeBufferPointer<Double>) -> Double {
        vDSP.sum(vector)
    }

    public static func _vDSPMean(_ vector: UnsafeBufferPointer<Double>) -> Double {
        vDSP.mean(vector)
    }

    public static func _vDSPDot(_ lhs: UnsafeBufferPointer<Double>, _ rhs: UnsafeBufferPointer<Double>) -> Double {
        var result: Double = 0
        vDSP_dotprD(lhs.baseAddress!, 1, rhs.baseAddress!, 1, &result, vDSP_Length(lhs.count))
        return result
    }

    public static func _cblasGemm(
        m: Int32, n: Int32, k: Int32,
        a: UnsafePointer<Double>, lda: Int32,
        b: UnsafePointer<Double>, ldb: Int32,
        c: UnsafeMutablePointer<Double>, ldc: Int32
    ) {
        cblas_dgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            m, n, k,
            1.0, a, lda,
            b, ldb,
            0.0, c, ldc
        )
    }
}

// MARK: - Contiguous element helper

extension Tensor {
    /// Returns a contiguous `Array` of the tensor's logical elements.
    ///
    /// If the tensor is already contiguous, returns `storage` directly (no copy).
    /// Otherwise, materializes the elements by iterating in row-major order.
    func contiguousElements() -> [Element] {
        if isContiguous {
            return storage
        }
        return Array(self)
    }
}

// MARK: - Accelerate-optimized reductions

extension Tensor where Element: AccelerateFloatingPoint {

    /// Returns the sum of all elements using vDSP.
    public func sum() -> Element {
        let elements = contiguousElements()
        return elements.withUnsafeBufferPointer { Element._vDSPSum($0) }
    }

    /// Returns the arithmetic mean of all elements using vDSP.
    public func mean() -> Element {
        let elements = contiguousElements()
        return elements.withUnsafeBufferPointer { Element._vDSPMean($0) }
    }

    /// Computes the dot product of two rank-1 tensors using vDSP.
    public static func dot(_ lhs: Tensor, _ rhs: Tensor) -> Element {
        precondition(lhs.rank == 1 && rhs.rank == 1,
                     "dot requires rank-1 tensors, got ranks \(lhs.rank) and \(rhs.rank)")
        precondition(lhs.shape == rhs.shape,
                     "Shape mismatch: \(lhs.shape) vs \(rhs.shape)")
        let lhsElements = lhs.contiguousElements()
        let rhsElements = rhs.contiguousElements()
        return lhsElements.withUnsafeBufferPointer { lBuf in
            rhsElements.withUnsafeBufferPointer { rBuf in
                Element._vDSPDot(lBuf, rBuf)
            }
        }
    }

    /// Multiplies two rank-2 tensors using CBLAS gemm.
    public static func matmul(_ lhs: Tensor, _ rhs: Tensor) -> Tensor {
        precondition(lhs.rank == 2 && rhs.rank == 2,
                     "matmul requires rank-2 tensors, got ranks \(lhs.rank) and \(rhs.rank)")
        let m = lhs.shape[0]
        let inner = lhs.shape[1]
        let n = rhs.shape[1]
        precondition(inner == rhs.shape[0],
                     "Inner dimensions must match: \(lhs.shape) vs \(rhs.shape)")
        let lhsElements = lhs.contiguousElements()
        let rhsElements = rhs.contiguousElements()
        var result = [Element](repeating: .zero, count: m * n)
        lhsElements.withUnsafeBufferPointer { aBuf in
            rhsElements.withUnsafeBufferPointer { bBuf in
                result.withUnsafeMutableBufferPointer { cBuf in
                    Element._cblasGemm(
                        m: Int32(m), n: Int32(n), k: Int32(inner),
                        a: aBuf.baseAddress!, lda: Int32(inner),
                        b: bBuf.baseAddress!, ldb: Int32(n),
                        c: cBuf.baseAddress!, ldc: Int32(n)
                    )
                }
            }
        }
        return Tensor(shape: [m, n], elements: result)
    }

    /// Returns a tensor with one axis collapsed by summation.
    ///
    /// Uses vDSP for rank-2 tensors; falls back to the generic path for higher ranks.
    public func sum(axis: Int) -> Tensor {
        precondition(axis >= 0 && axis < rank,
                     "Axis \(axis) out of range for rank \(rank)")
        guard rank == 2 else { return _sumAxis(axis) }

        let rows = shape[0]
        let cols = shape[1]
        let elements = contiguousElements()

        if axis == 0 {
            // Sum across rows -> shape [cols]
            var result = [Element](repeating: .zero, count: cols)
            elements.withUnsafeBufferPointer { buf in
                for i in 0..<rows {
                    let rowBuf = UnsafeBufferPointer(
                        start: buf.baseAddress! + i * cols,
                        count: cols
                    )
                    result.withUnsafeMutableBufferPointer { resBuf in
                        for j in 0..<cols {
                            resBuf[j] += rowBuf[j]
                        }
                    }
                }
            }
            return Tensor(shape: [cols], elements: result)
        } else {
            // axis == 1: Sum across columns -> shape [rows]
            var result = [Element](repeating: .zero, count: rows)
            elements.withUnsafeBufferPointer { buf in
                for i in 0..<rows {
                    let rowBuf = UnsafeBufferPointer(
                        start: buf.baseAddress! + i * cols,
                        count: cols
                    )
                    result[i] = Element._vDSPSum(rowBuf)
                }
            }
            return Tensor(shape: [rows], elements: result)
        }
    }

    /// Returns a tensor with one axis collapsed by averaging.
    ///
    /// Uses the Accelerate `sum(axis:)` for rank-2 tensors; falls back to the generic
    /// path for higher ranks.
    public func mean(axis: Int) -> Tensor {
        precondition(axis >= 0 && axis < rank,
                     "Axis \(axis) out of range for rank \(rank)")
        guard rank == 2 else { return _meanAxis(axis) }
        let s = sum(axis: axis)
        let divisor = Element(shape[axis])
        return Tensor(shape: s.shape, elements: s.map { $0 / divisor })
    }
}

#endif

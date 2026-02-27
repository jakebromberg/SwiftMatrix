#if canImport(Accelerate)

import Accelerate

/// Abstracts vDSP/CBLAS operations over `Float` and `Double` so
/// Accelerate-optimized `Tensor` extensions can be written once.
///
/// Only `Float` and `Double` conform. The protocol is public so that
/// overload resolution can prefer the Accelerate specializations over
/// the generic `AdditiveArithmetic`/`Numeric`/`FloatingPoint` versions.
public protocol AccelerateFloatingPoint: FloatingPoint {
    // Reductions
    static func _vDSPSum(_ vector: UnsafeBufferPointer<Self>) -> Self
    static func _vDSPMean(_ vector: UnsafeBufferPointer<Self>) -> Self
    static func _vDSPDot(_ lhs: UnsafeBufferPointer<Self>, _ rhs: UnsafeBufferPointer<Self>) -> Self
    static func _cblasGemm(
        m: Int32, n: Int32, k: Int32,
        a: UnsafePointer<Self>, lda: Int32,
        b: UnsafePointer<Self>, ldb: Int32,
        c: UnsafeMutablePointer<Self>, ldc: Int32
    )

    // Element-wise arithmetic
    static func _vDSPAdd(
        _ a: UnsafeBufferPointer<Self>, _ b: UnsafeBufferPointer<Self>,
        result: UnsafeMutableBufferPointer<Self>)
    static func _vDSPSub(
        _ lhs: UnsafeBufferPointer<Self>, _ rhs: UnsafeBufferPointer<Self>,
        result: UnsafeMutableBufferPointer<Self>)
    static func _vDSPMul(
        _ a: UnsafeBufferPointer<Self>, _ b: UnsafeBufferPointer<Self>,
        result: UnsafeMutableBufferPointer<Self>)
    static func _vDSPDiv(
        _ lhs: UnsafeBufferPointer<Self>, _ rhs: UnsafeBufferPointer<Self>,
        result: UnsafeMutableBufferPointer<Self>)
    static func _vDSPScalarAdd(
        _ vector: UnsafeBufferPointer<Self>, _ scalar: Self,
        result: UnsafeMutableBufferPointer<Self>)
    static func _vDSPScalarMul(
        _ vector: UnsafeBufferPointer<Self>, _ scalar: Self,
        result: UnsafeMutableBufferPointer<Self>)
    static func _vDSPNeg(
        _ vector: UnsafeBufferPointer<Self>,
        result: UnsafeMutableBufferPointer<Self>)
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

    public static func _vDSPAdd(
        _ a: UnsafeBufferPointer<Float>, _ b: UnsafeBufferPointer<Float>,
        result: UnsafeMutableBufferPointer<Float>
    ) {
        vDSP_vadd(a.baseAddress!, 1, b.baseAddress!, 1,
                  result.baseAddress!, 1, vDSP_Length(a.count))
    }

    public static func _vDSPSub(
        _ lhs: UnsafeBufferPointer<Float>, _ rhs: UnsafeBufferPointer<Float>,
        result: UnsafeMutableBufferPointer<Float>
    ) {
        // vDSP_vsub computes B - A, so pass rhs as A and lhs as B to get lhs - rhs
        vDSP_vsub(rhs.baseAddress!, 1, lhs.baseAddress!, 1,
                  result.baseAddress!, 1, vDSP_Length(lhs.count))
    }

    public static func _vDSPMul(
        _ a: UnsafeBufferPointer<Float>, _ b: UnsafeBufferPointer<Float>,
        result: UnsafeMutableBufferPointer<Float>
    ) {
        vDSP_vmul(a.baseAddress!, 1, b.baseAddress!, 1,
                  result.baseAddress!, 1, vDSP_Length(a.count))
    }

    public static func _vDSPDiv(
        _ lhs: UnsafeBufferPointer<Float>, _ rhs: UnsafeBufferPointer<Float>,
        result: UnsafeMutableBufferPointer<Float>
    ) {
        // vDSP_vdiv computes B / A, so pass rhs as A and lhs as B to get lhs / rhs
        vDSP_vdiv(rhs.baseAddress!, 1, lhs.baseAddress!, 1,
                  result.baseAddress!, 1, vDSP_Length(lhs.count))
    }

    public static func _vDSPScalarAdd(
        _ vector: UnsafeBufferPointer<Float>, _ scalar: Float,
        result: UnsafeMutableBufferPointer<Float>
    ) {
        var s = scalar
        vDSP_vsadd(vector.baseAddress!, 1, &s,
                   result.baseAddress!, 1, vDSP_Length(vector.count))
    }

    public static func _vDSPScalarMul(
        _ vector: UnsafeBufferPointer<Float>, _ scalar: Float,
        result: UnsafeMutableBufferPointer<Float>
    ) {
        var s = scalar
        vDSP_vsmul(vector.baseAddress!, 1, &s,
                   result.baseAddress!, 1, vDSP_Length(vector.count))
    }

    public static func _vDSPNeg(
        _ vector: UnsafeBufferPointer<Float>,
        result: UnsafeMutableBufferPointer<Float>
    ) {
        vDSP_vneg(vector.baseAddress!, 1,
                  result.baseAddress!, 1, vDSP_Length(vector.count))
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

    public static func _vDSPAdd(
        _ a: UnsafeBufferPointer<Double>, _ b: UnsafeBufferPointer<Double>,
        result: UnsafeMutableBufferPointer<Double>
    ) {
        vDSP_vaddD(a.baseAddress!, 1, b.baseAddress!, 1,
                   result.baseAddress!, 1, vDSP_Length(a.count))
    }

    public static func _vDSPSub(
        _ lhs: UnsafeBufferPointer<Double>, _ rhs: UnsafeBufferPointer<Double>,
        result: UnsafeMutableBufferPointer<Double>
    ) {
        vDSP_vsubD(rhs.baseAddress!, 1, lhs.baseAddress!, 1,
                   result.baseAddress!, 1, vDSP_Length(lhs.count))
    }

    public static func _vDSPMul(
        _ a: UnsafeBufferPointer<Double>, _ b: UnsafeBufferPointer<Double>,
        result: UnsafeMutableBufferPointer<Double>
    ) {
        vDSP_vmulD(a.baseAddress!, 1, b.baseAddress!, 1,
                   result.baseAddress!, 1, vDSP_Length(a.count))
    }

    public static func _vDSPDiv(
        _ lhs: UnsafeBufferPointer<Double>, _ rhs: UnsafeBufferPointer<Double>,
        result: UnsafeMutableBufferPointer<Double>
    ) {
        vDSP_vdivD(rhs.baseAddress!, 1, lhs.baseAddress!, 1,
                   result.baseAddress!, 1, vDSP_Length(lhs.count))
    }

    public static func _vDSPScalarAdd(
        _ vector: UnsafeBufferPointer<Double>, _ scalar: Double,
        result: UnsafeMutableBufferPointer<Double>
    ) {
        var s = scalar
        vDSP_vsaddD(vector.baseAddress!, 1, &s,
                    result.baseAddress!, 1, vDSP_Length(vector.count))
    }

    public static func _vDSPScalarMul(
        _ vector: UnsafeBufferPointer<Double>, _ scalar: Double,
        result: UnsafeMutableBufferPointer<Double>
    ) {
        var s = scalar
        vDSP_vsmulD(vector.baseAddress!, 1, &s,
                    result.baseAddress!, 1, vDSP_Length(vector.count))
    }

    public static func _vDSPNeg(
        _ vector: UnsafeBufferPointer<Double>,
        result: UnsafeMutableBufferPointer<Double>
    ) {
        vDSP_vnegD(vector.baseAddress!, 1,
                   result.baseAddress!, 1, vDSP_Length(vector.count))
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

// MARK: - vDSP element-wise binary helpers

/// Applies a vDSP binary operation to two tensors, materializing non-contiguous inputs as needed.
/// When shapes differ, broadcasts both to the output shape before applying the operation.
private func accelerateElementwise<T: AccelerateFloatingPoint>(
    _ lhs: Tensor<T>, _ rhs: Tensor<T>,
    body: (UnsafeBufferPointer<T>, UnsafeBufferPointer<T>, UnsafeMutableBufferPointer<T>) -> Void
) -> Tensor<T> {
    let outputShape: [Int]
    let lOp: Tensor<T>
    let rOp: Tensor<T>
    if lhs.shape == rhs.shape {
        outputShape = lhs.shape
        lOp = lhs
        rOp = rhs
    } else {
        guard let bShape = Tensor<T>.broadcastShape(lhs.shape, rhs.shape) else {
            preconditionFailure("Cannot broadcast shapes \(lhs.shape) and \(rhs.shape)")
        }
        outputShape = bShape
        lOp = lhs.broadcast(to: bShape)
        rOp = rhs.broadcast(to: bShape)
    }
    let lhsElements = lOp.contiguousElements()
    let rhsElements = rOp.contiguousElements()
    var result = [T](repeating: .zero, count: outputShape.reduce(1, *))
    lhsElements.withUnsafeBufferPointer { lBuf in
        rhsElements.withUnsafeBufferPointer { rBuf in
            result.withUnsafeMutableBufferPointer { resBuf in
                body(lBuf, rBuf, resBuf)
            }
        }
    }
    return Tensor<T>(shape: outputShape, elements: result)
}

/// Applies a vDSP unary operation to a tensor.
private func accelerateUnary<T: AccelerateFloatingPoint>(
    _ operand: Tensor<T>,
    body: (UnsafeBufferPointer<T>, UnsafeMutableBufferPointer<T>) -> Void
) -> Tensor<T> {
    let elements = operand.contiguousElements()
    var result = [T](repeating: .zero, count: operand.count)
    elements.withUnsafeBufferPointer { inBuf in
        result.withUnsafeMutableBufferPointer { outBuf in
            body(inBuf, outBuf)
        }
    }
    return Tensor<T>(shape: operand.shape, elements: result)
}

// MARK: - Accelerate-optimized element-wise arithmetic

extension Tensor where Element: AccelerateFloatingPoint {

    // MARK: Tensor + Tensor

    public static func + (lhs: Tensor, rhs: Tensor) -> Tensor {
        accelerateElementwise(lhs, rhs) { Element._vDSPAdd($0, $1, result: $2) }
    }

    public static func - (lhs: Tensor, rhs: Tensor) -> Tensor {
        accelerateElementwise(lhs, rhs) { Element._vDSPSub($0, $1, result: $2) }
    }

    public static func * (lhs: Tensor, rhs: Tensor) -> Tensor {
        accelerateElementwise(lhs, rhs) { Element._vDSPMul($0, $1, result: $2) }
    }

    public static func / (lhs: Tensor, rhs: Tensor) -> Tensor {
        accelerateElementwise(lhs, rhs) { Element._vDSPDiv($0, $1, result: $2) }
    }

    // MARK: Negation

    public static prefix func - (operand: Tensor) -> Tensor {
        accelerateUnary(operand) { Element._vDSPNeg($0, result: $1) }
    }

    // MARK: Tensor + Scalar / Scalar + Tensor

    public static func + (lhs: Tensor, rhs: Element) -> Tensor {
        accelerateUnary(lhs) { Element._vDSPScalarAdd($0, rhs, result: $1) }
    }

    public static func + (lhs: Element, rhs: Tensor) -> Tensor {
        accelerateUnary(rhs) { Element._vDSPScalarAdd($0, lhs, result: $1) }
    }

    public static func - (lhs: Tensor, rhs: Element) -> Tensor {
        accelerateUnary(lhs) { Element._vDSPScalarAdd($0, -rhs, result: $1) }
    }

    public static func - (lhs: Element, rhs: Tensor) -> Tensor {
        let negated = accelerateUnary(rhs) { Element._vDSPNeg($0, result: $1) }
        return accelerateUnary(negated) { Element._vDSPScalarAdd($0, lhs, result: $1) }
    }

    public static func * (lhs: Tensor, rhs: Element) -> Tensor {
        accelerateUnary(lhs) { Element._vDSPScalarMul($0, rhs, result: $1) }
    }

    public static func * (lhs: Element, rhs: Tensor) -> Tensor {
        accelerateUnary(rhs) { Element._vDSPScalarMul($0, lhs, result: $1) }
    }

    public static func / (lhs: Tensor, rhs: Element) -> Tensor {
        accelerateUnary(lhs) { Element._vDSPScalarMul($0, 1 / rhs, result: $1) }
    }

    // MARK: Compound assignment

    public static func += (lhs: inout Tensor, rhs: Tensor) {
        lhs = lhs + rhs
    }

    public static func -= (lhs: inout Tensor, rhs: Tensor) {
        lhs = lhs - rhs
    }

    public static func *= (lhs: inout Tensor, rhs: Tensor) {
        lhs = lhs * rhs
    }

    public static func /= (lhs: inout Tensor, rhs: Tensor) {
        lhs = lhs / rhs
    }
}

#endif

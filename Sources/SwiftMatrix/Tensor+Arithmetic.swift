/// Element-wise arithmetic for tensors with matching shapes.
///
/// All operations require operands to have the same shape and produce a new contiguous tensor.
/// No broadcasting is performed. When both operands are contiguous, the storage arrays are
/// zipped directly, bypassing the `storageIndex(forLinearIndex:)` indirection.

private func elementwise<T>(
    _ lhs: Tensor<T>, _ rhs: Tensor<T>, body: (T, T) -> T
) -> Tensor<T> {
    precondition(lhs.shape == rhs.shape,
                 "Shape mismatch: \(lhs.shape) vs \(rhs.shape)")
    if lhs.isContiguous && rhs.isContiguous {
        return Tensor<T>(shape: lhs.shape,
                         elements: zip(lhs.storage, rhs.storage).map(body))
    }
    return Tensor<T>(shape: lhs.shape, elements: zip(lhs, rhs).map(body))
}

private func elementwiseInPlace<T>(
    _ lhs: inout Tensor<T>, _ rhs: Tensor<T>, body: (inout T, T) -> Void
) {
    precondition(lhs.shape == rhs.shape,
                 "Shape mismatch: \(lhs.shape) vs \(rhs.shape)")
    if lhs.isContiguous && rhs.isContiguous {
        for i in 0..<lhs.count {
            body(&lhs.storage[i], rhs.storage[i])
        }
        return
    }
    lhs = elementwise(lhs, rhs) { var a = $0; body(&a, $1); return a }
}

// MARK: - Tensor + Tensor

extension Tensor where Element: AdditiveArithmetic {
    public static func + (lhs: Tensor, rhs: Tensor) -> Tensor {
        elementwise(lhs, rhs, body: +)
    }

    public static func - (lhs: Tensor, rhs: Tensor) -> Tensor {
        elementwise(lhs, rhs, body: -)
    }
}

extension Tensor where Element: Numeric {
    public static func * (lhs: Tensor, rhs: Tensor) -> Tensor {
        elementwise(lhs, rhs, body: *)
    }
}

extension Tensor where Element: FloatingPoint {
    public static func / (lhs: Tensor, rhs: Tensor) -> Tensor {
        elementwise(lhs, rhs, body: /)
    }
}

extension Tensor where Element: SignedNumeric {
    public static prefix func - (operand: Tensor) -> Tensor {
        if operand.isContiguous {
            return Tensor(shape: operand.shape, elements: operand.storage.map { -$0 })
        }
        return Tensor(shape: operand.shape, elements: operand.map { -$0 })
    }
}

// MARK: - Tensor + Scalar / Scalar + Tensor

extension Tensor where Element: AdditiveArithmetic {
    public static func + (lhs: Tensor, rhs: Element) -> Tensor {
        if lhs.isContiguous {
            return Tensor(shape: lhs.shape, elements: lhs.storage.map { $0 + rhs })
        }
        return Tensor(shape: lhs.shape, elements: lhs.map { $0 + rhs })
    }

    public static func + (lhs: Element, rhs: Tensor) -> Tensor {
        if rhs.isContiguous {
            return Tensor(shape: rhs.shape, elements: rhs.storage.map { lhs + $0 })
        }
        return Tensor(shape: rhs.shape, elements: rhs.map { lhs + $0 })
    }

    public static func - (lhs: Tensor, rhs: Element) -> Tensor {
        if lhs.isContiguous {
            return Tensor(shape: lhs.shape, elements: lhs.storage.map { $0 - rhs })
        }
        return Tensor(shape: lhs.shape, elements: lhs.map { $0 - rhs })
    }

    public static func - (lhs: Element, rhs: Tensor) -> Tensor {
        if rhs.isContiguous {
            return Tensor(shape: rhs.shape, elements: rhs.storage.map { lhs - $0 })
        }
        return Tensor(shape: rhs.shape, elements: rhs.map { lhs - $0 })
    }
}

extension Tensor where Element: Numeric {
    public static func * (lhs: Tensor, rhs: Element) -> Tensor {
        if lhs.isContiguous {
            return Tensor(shape: lhs.shape, elements: lhs.storage.map { $0 * rhs })
        }
        return Tensor(shape: lhs.shape, elements: lhs.map { $0 * rhs })
    }

    public static func * (lhs: Element, rhs: Tensor) -> Tensor {
        if rhs.isContiguous {
            return Tensor(shape: rhs.shape, elements: rhs.storage.map { lhs * $0 })
        }
        return Tensor(shape: rhs.shape, elements: rhs.map { lhs * $0 })
    }
}

extension Tensor where Element: FloatingPoint {
    public static func / (lhs: Tensor, rhs: Element) -> Tensor {
        if lhs.isContiguous {
            return Tensor(shape: lhs.shape, elements: lhs.storage.map { $0 / rhs })
        }
        return Tensor(shape: lhs.shape, elements: lhs.map { $0 / rhs })
    }
}

// MARK: - Compound assignment

extension Tensor where Element: AdditiveArithmetic {
    public static func += (lhs: inout Tensor, rhs: Tensor) {
        elementwiseInPlace(&lhs, rhs) { $0 += $1 }
    }

    public static func -= (lhs: inout Tensor, rhs: Tensor) {
        elementwiseInPlace(&lhs, rhs) { $0 -= $1 }
    }
}

extension Tensor where Element: Numeric {
    public static func *= (lhs: inout Tensor, rhs: Tensor) {
        elementwiseInPlace(&lhs, rhs) { $0 *= $1 }
    }
}

extension Tensor where Element: FloatingPoint {
    public static func /= (lhs: inout Tensor, rhs: Tensor) {
        elementwiseInPlace(&lhs, rhs) { $0 /= $1 }
    }
}

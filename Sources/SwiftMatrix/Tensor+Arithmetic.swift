/// Element-wise arithmetic for tensors with matching shapes.
///
/// All operations require operands to have the same shape and produce a new contiguous tensor.
/// No broadcasting is performed.

private func elementwise<T>(
    _ lhs: Tensor<T>, _ rhs: Tensor<T>, body: (T, T) -> T
) -> Tensor<T> {
    precondition(lhs.shape == rhs.shape,
                 "Shape mismatch: \(lhs.shape) vs \(rhs.shape)")
    return Tensor<T>(shape: lhs.shape, elements: zip(lhs, rhs).map(body))
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
        Tensor(shape: operand.shape, elements: operand.map { -$0 })
    }
}

// MARK: - Tensor + Scalar / Scalar + Tensor

extension Tensor where Element: AdditiveArithmetic {
    public static func + (lhs: Tensor, rhs: Element) -> Tensor {
        Tensor(shape: lhs.shape, elements: lhs.map { $0 + rhs })
    }

    public static func + (lhs: Element, rhs: Tensor) -> Tensor {
        Tensor(shape: rhs.shape, elements: rhs.map { lhs + $0 })
    }

    public static func - (lhs: Tensor, rhs: Element) -> Tensor {
        Tensor(shape: lhs.shape, elements: lhs.map { $0 - rhs })
    }

    public static func - (lhs: Element, rhs: Tensor) -> Tensor {
        Tensor(shape: rhs.shape, elements: rhs.map { lhs - $0 })
    }
}

extension Tensor where Element: Numeric {
    public static func * (lhs: Tensor, rhs: Element) -> Tensor {
        Tensor(shape: lhs.shape, elements: lhs.map { $0 * rhs })
    }

    public static func * (lhs: Element, rhs: Tensor) -> Tensor {
        Tensor(shape: rhs.shape, elements: rhs.map { lhs * $0 })
    }
}

extension Tensor where Element: FloatingPoint {
    public static func / (lhs: Tensor, rhs: Element) -> Tensor {
        Tensor(shape: lhs.shape, elements: lhs.map { $0 / rhs })
    }
}

// MARK: - Compound assignment

extension Tensor where Element: AdditiveArithmetic {
    public static func += (lhs: inout Tensor, rhs: Tensor) {
        lhs = lhs + rhs
    }

    public static func -= (lhs: inout Tensor, rhs: Tensor) {
        lhs = lhs - rhs
    }
}

extension Tensor where Element: Numeric {
    public static func *= (lhs: inout Tensor, rhs: Tensor) {
        lhs = lhs * rhs
    }
}

extension Tensor where Element: FloatingPoint {
    public static func /= (lhs: inout Tensor, rhs: Tensor) {
        lhs = lhs / rhs
    }
}

/// Lazy evaluation entry point and operators for ``TensorExpression``.
///
/// Use `Tensor.lazy` to enter the lazy world, compose operations, then materialize
/// with `Tensor(evaluating:)`.
///
/// ```swift
/// let result = Tensor(evaluating: a.lazy + b.lazy * c.lazy)
/// // One pass: result[i] = a[i] + b[i] * c[i], no intermediates
/// ```

extension Tensor {
    /// Returns a lazy wrapper for expression-based arithmetic.
    ///
    /// This shadows `Sequence.lazy` intentionally. The returned `LazyTensor` conforms to
    /// `TensorExpression` and participates in lazy operator overload resolution.
    public var lazy: LazyTensor<Element> {
        LazyTensor(tensor: self)
    }

    /// Materializes a lazy expression into a contiguous `Tensor`.
    ///
    /// - Parameter expression: A `TensorExpression` tree to evaluate.
    public init<E: TensorExpression>(evaluating expression: E) where E.Element == Element {
        self = expression.evaluate()
    }
}

// MARK: - Lazy binary operators

extension TensorExpression where Element: AdditiveArithmetic {
    public static func + <RHS: TensorExpression>(
        lhs: Self, rhs: RHS
    ) -> BinaryExpression<Self, RHS> where RHS.Element == Element {
        BinaryExpression(lhs: lhs, rhs: rhs, op: +, shape: lhs.shape)
    }

    public static func - <RHS: TensorExpression>(
        lhs: Self, rhs: RHS
    ) -> BinaryExpression<Self, RHS> where RHS.Element == Element {
        BinaryExpression(lhs: lhs, rhs: rhs, op: -, shape: lhs.shape)
    }
}

extension TensorExpression where Element: Numeric {
    public static func * <RHS: TensorExpression>(
        lhs: Self, rhs: RHS
    ) -> BinaryExpression<Self, RHS> where RHS.Element == Element {
        BinaryExpression(lhs: lhs, rhs: rhs, op: *, shape: lhs.shape)
    }
}

extension TensorExpression where Element: FloatingPoint {
    public static func / <RHS: TensorExpression>(
        lhs: Self, rhs: RHS
    ) -> BinaryExpression<Self, RHS> where RHS.Element == Element {
        BinaryExpression(lhs: lhs, rhs: rhs, op: /, shape: lhs.shape)
    }
}

// MARK: - Lazy unary operators

extension TensorExpression where Element: SignedNumeric {
    public static prefix func - (operand: Self) -> UnaryExpression<Self> {
        UnaryExpression(expr: operand) { -$0 }
    }
}

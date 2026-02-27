/// A lazily-evaluated tensor expression.
///
/// Types conforming to `TensorExpression` represent deferred computations that can be
/// composed into expression trees. Evaluation only occurs when `evaluate()` is called,
/// fusing all operations into a single pass with zero intermediate allocations.
///
/// `Tensor` does **not** conform to this protocol, cleanly separating the eager and lazy
/// worlds. Users opt into lazy evaluation via `Tensor.lazy`.
public protocol TensorExpression<Element> {
    associatedtype Element
    var shape: [Int] { get }
    var count: Int { get }
    func element(at linearIndex: Int) -> Element
}

extension TensorExpression {
    /// Materializes the expression into a contiguous `Tensor`.
    ///
    /// Iterates `0..<count` calling `element(at:)` -- the recursion through the expression
    /// tree fuses all operations into a single pass per element with zero intermediate
    /// allocations.
    public func evaluate() -> Tensor<Element> {
        var elements = [Element]()
        elements.reserveCapacity(count)
        for i in 0..<count {
            elements.append(element(at: i))
        }
        return Tensor(shape: shape, elements: elements)
    }
}

/// Wraps a `Tensor` for lazy expression arithmetic.
///
/// Created via `Tensor.lazy`. Reads elements through the tensor's `storageIndex` machinery,
/// so non-contiguous inputs (transposed, sliced) work correctly.
public struct LazyTensor<Element>: TensorExpression {
    let tensor: Tensor<Element>

    public var shape: [Int] { tensor.shape }
    public var count: Int { tensor.count }

    public func element(at linearIndex: Int) -> Element {
        tensor.storage[tensor.storageIndex(forLinearIndex: linearIndex)]
    }
}

/// Element-wise binary operation (deferred).
///
/// Stores references to both operands and the operation closure. On `element(at:)`,
/// evaluates both operands at the same index and applies the operation.
public struct BinaryExpression<LHS: TensorExpression, RHS: TensorExpression>: TensorExpression
where LHS.Element == RHS.Element {
    public typealias Element = LHS.Element

    let lhs: LHS
    let rhs: RHS
    let op: (Element, Element) -> Element

    public let shape: [Int]
    public var count: Int { shape.reduce(1, *) }

    public func element(at i: Int) -> Element {
        op(lhs.element(at: i), rhs.element(at: i))
    }
}

/// Element-wise unary operation (deferred).
///
/// Stores a reference to the operand and the operation closure.
public struct UnaryExpression<Expr: TensorExpression>: TensorExpression {
    public typealias Element = Expr.Element

    let expr: Expr
    let op: (Element) -> Element

    public var shape: [Int] { expr.shape }
    public var count: Int { expr.count }

    public func element(at i: Int) -> Element {
        op(expr.element(at: i))
    }
}

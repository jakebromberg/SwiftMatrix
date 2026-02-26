/// A two-component integer index used by ``Matrix`` for row/column addressing.
///
/// `Pair` serves as the `Collection.Index` type for ``Matrix``. The `x` component addresses
/// rows and `y` addresses columns.
///
/// ```swift
/// let p = Pair(x: 1, y: 2)
/// matrix[p]  // element at row 1, column 2
/// ```
///
/// `Pair` conforms to `Comparable` using a component-wise partial order (both `x` and `y`
/// must be less). This is sufficient for ``Matrix``'s `Collection` conformance but does not
/// define a total order over all pairs.
public struct Pair: Comparable {
  public static func < (lhs: Pair, rhs: Pair) -> Bool {
    lhs.x < rhs.x && lhs.y < rhs.y
  }

  /// The row index.
  public let x: Int

  /// The column index.
  public let y: Int

  public init(x: Int, y: Int) {
    self.x = x
    self.y = y
  }

  /// Returns the component-wise sum of two pairs.
  public static func +(lhs: Pair, rhs: Pair) -> Pair {
    Pair(x: lhs.x + rhs.x, y: lhs.y + rhs.y)
  }

  /// Returns the component-wise remainder of a pair divided by a vector's dimensions.
  ///
  /// Wraps `x` by ``Vector/width`` and `y` by ``Vector/depth``. Used by
  /// ``Matrix/offset(by:)`` for toroidal index wrapping.
  public static func %(lhs: Pair, rhs: Vector) -> Pair {
    Pair(x: lhs.x % rhs.width, y: lhs.y % rhs.depth)
  }
}

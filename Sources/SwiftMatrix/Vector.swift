/// A two-component size descriptor used by ``Matrix``.
///
/// `Vector` describes the dimensions of a ``Matrix``, with ``width`` for the number of
/// columns and ``depth`` for the number of rows.
///
/// ```swift
/// let size = Vector(width: 3, depth: 2)  // 2 rows, 3 columns
/// let m = Matrix(size: size, repeating: 0)
/// ```
public struct Vector: Hashable {
  /// The number of columns (axis 1 size).
  public let width: Int

  /// The number of rows (axis 0 size).
  public let depth: Int

  public init(width: Int, depth: Int) {
    self.width = width
    self.depth = depth
  }
}

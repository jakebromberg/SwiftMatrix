/// A rank-2 collection backed by nested arrays.
///
/// `Matrix` stores elements as `[[E]]` and conforms to `Collection` with ``Pair`` as its
/// index type. Elements are iterated in row-major order: all columns of row 0, then row 1,
/// and so on.
///
/// ```swift
/// let m = Matrix([[1, 2, 3],
///                 [4, 5, 6]])
/// m[Pair(x: 0, y: 2)]  // 3
/// Array(m)              // [1, 2, 3, 4, 5, 6]
/// ```
///
/// For new code, prefer ``Tensor`` which supports arbitrary rank and uses flat storage
/// with shape/strides for efficient zero-copy operations.
public struct Matrix<E> {
  fileprivate let storage: [[E]]

  /// The dimensions of this matrix.
  public let size: Vector

  /// Creates a matrix filled with a repeated value.
  ///
  /// - Parameters:
  ///   - size: The dimensions of the matrix (`width` columns, `depth` rows).
  ///   - repeatedValue: The value to fill every element with.
  public init(size: Vector, repeating repeatedValue: E) {
    let row: [E] = Array(repeating: repeatedValue, count: size.width)
    self.storage = Array(repeating: row, count: size.depth)
    self.size = size
  }

  /// Creates a matrix from nested arrays.
  ///
  /// The outer array represents rows and each inner array represents columns.
  /// All inner arrays must have the same length.
  ///
  /// - Parameter arrays: The rows of the matrix. All rows must have equal length.
  public init(_ arrays: [[E]]) {
    let widths = Set(arrays.map(\.count))

    guard widths.count == 1 else {
      fatalError()
    }

    self.storage = arrays
    self.size = Vector(width: widths.first!, depth: arrays.count)
  }
}

// MARK: - Collection

extension Matrix: Collection {
  public func index(after i: Pair) -> Pair {
    guard let width = storage.first?.count else {
      return Pair(x: 0, y: 0)
    }

    if i.y == width - 1 {
      return Pair(x: i.x + 1, y: 0)
    }

    return Pair(x: i.x, y: i.y + 1)
  }

  /// Accesses the element at the given row/column position.
  ///
  /// - Parameter position: A ``Pair`` where `x` is the row and `y` is the column.
  public subscript(position: Pair) -> E {
    self.storage[position.x][position.y]
  }

  public var startIndex: Pair {
    Pair(x: 0, y: 0)
  }

  public var endIndex: Pair {
    return Pair(x: storage.count, y: 0)
  }

  public typealias Index = Pair
  public typealias Element = E
}

extension Matrix: Equatable where Element: Equatable {

}

extension Matrix: Hashable where Element: Hashable {

}

// MARK: - Views

extension Matrix {
  /// Returns a view that transposes row and column access.
  ///
  /// The returned ``CollectionView`` swaps the `x` and `y` components of each index,
  /// so reading `inverted[Pair(x: r, y: c)]` returns `self[Pair(x: c, y: r)]`.
  ///
  /// - Returns: A transposed view over this matrix.
  public func invert() -> CollectionView<Matrix> {
    let swizzler = { (index: Pair) in
      Pair(x: index.y, y: index.x)
    }

    let result: CollectionView<Matrix> =
      CollectionView(swizzler: swizzler, storage: self)
    return result
  }

  /// Returns a view with indices shifted by the given offset, wrapping around the matrix.
  ///
  /// Each index `(x, y)` is mapped to `((x + pair.x) % size.width, (y + pair.y) % size.depth)`,
  /// creating a toroidal (wrap-around) shift of the matrix contents.
  ///
  /// - Parameter pair: The amount to shift indices by along each axis.
  /// - Returns: A cyclically shifted view over this matrix.
  public func offset(by pair: Pair) -> CollectionView<Matrix> {
    let swizzler = { (index: Pair) in
      (index + pair) % size
    }

    let result: CollectionView<Matrix> =
      CollectionView(swizzler: swizzler, storage: self)
    return result
  }
}

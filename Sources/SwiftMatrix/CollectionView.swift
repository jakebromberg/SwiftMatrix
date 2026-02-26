/// A lazy view over a collection that remaps indices through a swizzler function.
///
/// `CollectionView` wraps an existing collection and intercepts subscript access, passing
/// each index through a ``Swizzler`` before forwarding to the underlying storage. This
/// enables operations like transpose and cyclic shift without copying data.
///
/// ```swift
/// let matrix = Matrix([[1, 2], [3, 4]])
/// let transposed = CollectionView(
///     swizzler: { Pair(x: $0.y, y: $0.x) },
///     storage: matrix
/// )
/// transposed[Pair(x: 0, y: 1)]  // matrix[Pair(x: 1, y: 0)] == 3
/// ```
///
/// The view inherits `startIndex`, `endIndex`, and `index(after:)` directly from the
/// underlying collection. Only subscript access is remapped.
public struct CollectionView<Collection: Swift.Collection>: Swift.Collection {
  /// A function that transforms a logical index into a storage index.
  public typealias Swizzler = (Collection.Index) -> Collection.Index

  let swizzler: Swizzler
  let storage: Collection

  /// Creates a view that remaps index access through the given swizzler.
  ///
  /// - Parameters:
  ///   - swizzler: A function mapping logical indices to physical indices in `storage`.
  ///   - storage: The underlying collection to read elements from.
  public init(swizzler: @escaping Swizzler, storage: Collection) {
    self.swizzler = swizzler
    self.storage = storage
  }

  // MARK: - Collection

  public typealias Index = Collection.Index
  public typealias Element = Collection.Element

  /// Accesses the element at the swizzled position.
  ///
  /// The given `position` is passed through the ``Swizzler`` before indexing into the
  /// underlying storage.
  public subscript(position: Collection.Index) -> Collection.Element {
    storage[swizzler(position)]
  }

  public var startIndex: Collection.Index {
    storage.startIndex
  }

  public var endIndex: Collection.Index {
    storage.endIndex
  }

  public func index(after i: Collection.Index) -> Collection.Index {
    storage.index(after: i)
  }
}

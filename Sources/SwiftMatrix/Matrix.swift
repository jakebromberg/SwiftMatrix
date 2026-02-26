//
//  Matrix.swift
//  Sandbox
//
//  Created by Jake Bromberg on 4/22/21.
//  Copyright © 2021 Jake Bromberg. All rights reserved.
//

public struct Matrix<E> {
  fileprivate let storage: [[E]]

  public let size: Vector

  public init(size: Vector, repeating repeatedValue: E) {
    let row: [E] = Array(repeating: repeatedValue, count: size.width)
    self.storage = Array(repeating: row, count: size.depth)
    self.size = size
  }

  public init(_ arrays: [[E]]) {
    let widths = Set(arrays.map(\.count))

    guard widths.count == 1 else {
      fatalError()
    }

    self.storage = arrays
    self.size = Vector(width: widths.first!, depth: arrays.count)
  }
}

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

extension Matrix {
  public func invert() -> CollectionView<Matrix> {
    let swizzler = { (index: Pair) in
      Pair(x: index.y, y: index.x)
    }

    let result: CollectionView<Matrix> =
      CollectionView(swizzler: swizzler, storage: self)
    return result
  }

  public func offset(by pair: Pair) -> CollectionView<Matrix> {
    let swizzler = { (index: Pair) in
      (index + pair) % size
    }

    let result: CollectionView<Matrix> =
      CollectionView(swizzler: swizzler, storage: self)
    return result
  }
}

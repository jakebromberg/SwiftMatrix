//
//  Matrix.swift
//  Sandbox
//
//  Created by Jake Bromberg on 4/22/21.
//  Copyright © 2021 Jake Bromberg. All rights reserved.
//

import Foundation

struct Matrix<E> {
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
  func index(after i: Pair) -> Pair {
    guard let width = storage.first?.count else {
      return Pair(x: 0, y: 0)
    }
    
    if i.y == width - 1 {
      return Pair(x: i.x + 1, y: 0)
    }

    return Pair(x: i.x, y: i.y + 1)
  }
  
  subscript(position: Pair) -> E {
    self.storage[position.x][position.y]
  }
  
//  subscript(bounds: Range<Pair>) -> Slice<Matrix<E>> {
//    <#code#>
//  }
  
  var startIndex: Pair {
    Pair(x: 0, y: 0)
  }
  
  var endIndex: Pair {
    return Pair(x: storage.count, y: 0)
  }
  
  
  
  typealias Index = Pair
  typealias Element = E
}

extension Matrix: Equatable where Element: Equatable {
  
}

extension Matrix: Hashable where Element: Hashable {
  
}

extension Matrix {
  func invert() -> CollectionView<Matrix> {
    let swizzler = { (index: Pair) in
      Pair(x: index.y, y: index.x)
    }

    let result: CollectionView<Matrix> =
      CollectionView(swizzler: swizzler, storage: self)
    return result
  }
  
  func offset(by pair: Pair) -> CollectionView<Matrix> {
    let swizzler = { (index: Pair) in
      (index + pair) % size
    }
    
    let result: CollectionView<Matrix> =
      CollectionView(swizzler: swizzler, storage: self)
    return result
  }
}

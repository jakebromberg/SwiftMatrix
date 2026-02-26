//
//  CollectionView.swift
//  Sandbox
//
//  Created by Jake Bromberg on 4/22/21.
//  Copyright © 2021 Jake Bromberg. All rights reserved.
//

public struct CollectionView<Collection: Swift.Collection>: Swift.Collection {
  public typealias Swizzler = (Collection.Index) -> Collection.Index
  let swizzler: Swizzler
  let storage: Collection

  public init(swizzler: @escaping Swizzler, storage: Collection) {
    self.swizzler = swizzler
    self.storage = storage
  }

  // MARK: - Collection

  public typealias Index = Collection.Index
  public typealias Element = Collection.Element

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

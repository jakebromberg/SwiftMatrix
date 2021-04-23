//
//  CollectionView.swift
//  Sandbox
//
//  Created by Jake Bromberg on 4/22/21.
//  Copyright © 2021 Jake Bromberg. All rights reserved.
//

import Foundation

struct CollectionView<Collection: Swift.Collection>: Swift.Collection {
  typealias Swizzler = (Collection.Index) -> Collection.Index
  let swizzler: Swizzler
  let storage: Collection
  
  init(swizzler: @escaping Swizzler, storage: Collection) {
    self.swizzler = swizzler
    self.storage = storage
  }

  // MARK - Collection
  
  typealias Index = Collection.Index
  typealias Element = Collection.Element

  subscript(position: Collection.Index) -> Collection.Element {
    storage[swizzler(position)]
  }
  
  var startIndex: Collection.Index {
    storage.startIndex
  }
  
  var endIndex: Collection.Index {
    storage.endIndex
  }
  
  func index(after i: Collection.Index) -> Collection.Index {
    storage.index(after: i)
  }
}

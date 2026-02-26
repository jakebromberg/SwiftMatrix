//
//  Size.swift
//  Sandbox
//
//  Created by Jake Bromberg on 4/22/21.
//  Copyright © 2021 Jake Bromberg. All rights reserved.
//

public struct Vector: Hashable {
  public let width: Int
  public let depth: Int

  public init(width: Int, depth: Int) {
    self.width = width
    self.depth = depth
  }
}

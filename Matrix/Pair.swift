//
//  Pair.swift
//  Sandbox
//
//  Created by Jake Bromberg on 4/22/21.
//  Copyright © 2021 Jake Bromberg. All rights reserved.
//

import Foundation

public struct Pair: Comparable {
  public static func < (lhs: Pair, rhs: Pair) -> Bool {
    lhs.x < rhs.x && lhs.y < rhs.y
  }
  
  let x: Int
  let y: Int
  
  static func +(lhs: Pair, rhs: Pair) -> Pair {
    Pair(x: lhs.x + rhs.x, y: lhs.y + rhs.y)
  }

  static func %(lhs: Pair, rhs: Vector) -> Pair {
    Pair(x: lhs.x % rhs.width, y: lhs.y % rhs.depth)
  }
}

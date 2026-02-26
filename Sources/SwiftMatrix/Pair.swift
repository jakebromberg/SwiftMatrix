//
//  Pair.swift
//  Sandbox
//
//  Created by Jake Bromberg on 4/22/21.
//  Copyright © 2021 Jake Bromberg. All rights reserved.
//

public struct Pair: Comparable {
  public static func < (lhs: Pair, rhs: Pair) -> Bool {
    lhs.x < rhs.x && lhs.y < rhs.y
  }

  public let x: Int
  public let y: Int

  public init(x: Int, y: Int) {
    self.x = x
    self.y = y
  }

  public static func +(lhs: Pair, rhs: Pair) -> Pair {
    Pair(x: lhs.x + rhs.x, y: lhs.y + rhs.y)
  }

  public static func %(lhs: Pair, rhs: Vector) -> Pair {
    Pair(x: lhs.x % rhs.width, y: lhs.y % rhs.depth)
  }
}

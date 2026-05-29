//
//  Item.swift
//  KeyType
//
//  Created by John Bean on 5/29/26.
//

import Foundation
import SwiftData

@Model
final class Item {
    var timestamp: Date
    
    init(timestamp: Date) {
        self.timestamp = timestamp
    }
}

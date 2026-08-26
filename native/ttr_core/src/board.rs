//! Static board data, route definitions, and ticket indices for the USA map.

pub const USA_NUM_CITIES: usize = 36;
pub const USA_NUM_ROUTES: usize = 100;
pub const USA_NUM_TICKETS: usize = 30;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RouteDef {
    pub id: u8,
    pub city_a: u8,
    pub city_b: u8,
    pub length: u8,
    pub color: u8,                // 0..7 for colors, 255 for Gray
    pub double_pair_id: u8,       // Index of sibling route in double route, or 255 if single
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TicketDef {
    pub id: u8,
    pub city_a: u8,
    pub city_b: u8,
    pub points: u8,
}

pub static USA_CITY_NAMES: [&str; USA_NUM_CITIES] = [
    "Atlanta", "Boston", "Calgary", "Charleston", "Chicago", "Dallas", "Denver", "Duluth",
    "El Paso", "Helena", "Houston", "Kansas City", "Las Vegas", "Little Rock", "Los Angeles",
    "Miami", "Montreal", "Nashville", "New Orleans", "New York", "Oklahoma City", "Omaha",
    "Phoenix", "Pittsburgh", "Portland", "Raleigh", "Salt Lake City", "San Francisco",
    "Santa Fe", "Sault St. Marie", "Seattle", "St. Louis", "Toronto", "Vancouver",
    "Washington", "Winnipeg",
];

pub fn city_name_to_index(name: &str) -> Option<u8> {
    USA_CITY_NAMES.iter().position(|&c| c == name).map(|i| i as u8)
}

pub fn get_usa_routes() -> [RouteDef; USA_NUM_ROUTES] {
    // 100 USA routes matching python maps.py exact definition
    let raw: [(&str, &str, u8, &str); USA_NUM_ROUTES] = [
        ("Vancouver", "Calgary", 3, "X"),
        ("Vancouver", "Seattle", 1, "X"),
        ("Vancouver", "Seattle", 1, "X"),
        ("Seattle", "Calgary", 4, "X"),
        ("Seattle", "Helena", 6, "Y"),
        ("Seattle", "Portland", 1, "X"),
        ("Seattle", "Portland", 1, "X"),
        ("Portland", "Salt Lake City", 6, "B"),
        ("Portland", "San Francisco", 5, "G"),
        ("Portland", "San Francisco", 5, "P"),
        ("San Francisco", "Salt Lake City", 5, "O"),
        ("San Francisco", "Salt Lake City", 5, "W"),
        ("San Francisco", "Los Angeles", 3, "Y"),
        ("San Francisco", "Los Angeles", 3, "P"),
        ("Los Angeles", "Las Vegas", 2, "X"),
        ("Los Angeles", "Phoenix", 3, "X"),
        ("Los Angeles", "El Paso", 6, "K"),
        ("Calgary", "Winnipeg", 6, "W"),
        ("Calgary", "Helena", 4, "X"),
        ("Helena", "Winnipeg", 4, "B"),
        ("Helena", "Salt Lake City", 3, "P"),
        ("Helena", "Denver", 4, "G"),
        ("Helena", "Duluth", 6, "O"),
        ("Helena", "Omaha", 5, "R"),
        ("Salt Lake City", "Denver", 3, "R"),
        ("Salt Lake City", "Denver", 3, "Y"),
        ("Las Vegas", "Salt Lake City", 3, "O"),
        ("Phoenix", "Denver", 5, "W"),
        ("Phoenix", "Santa Fe", 3, "X"),
        ("Phoenix", "El Paso", 3, "X"),
        ("Winnipeg", "Sault St. Marie", 6, "X"),
        ("Winnipeg", "Duluth", 4, "K"),
        ("Duluth", "Sault St. Marie", 3, "X"),
        ("Duluth", "Toronto", 6, "P"),
        ("Duluth", "Chicago", 3, "R"),
        ("Duluth", "Omaha", 2, "X"),
        ("Duluth", "Omaha", 2, "X"),
        ("Omaha", "Chicago", 4, "B"),
        ("Omaha", "Kansas City", 1, "X"),
        ("Omaha", "Kansas City", 1, "X"),
        ("Kansas City", "St. Louis", 2, "B"),
        ("Kansas City", "St. Louis", 2, "P"),
        ("Kansas City", "Oklahoma City", 2, "X"),
        ("Kansas City", "Oklahoma City", 2, "X"),
        ("Oklahoma City", "Little Rock", 2, "X"),
        ("Oklahoma City", "Dallas", 2, "X"),
        ("Oklahoma City", "Dallas", 2, "X"),
        ("Dallas", "Little Rock", 2, "X"),
        ("Dallas", "Houston", 1, "X"),
        ("Dallas", "Houston", 1, "X"),
        ("Houston", "New Orleans", 2, "X"),
        ("El Paso", "Houston", 6, "G"),
        ("El Paso", "Dallas", 4, "R"),
        ("El Paso", "Oklahoma City", 5, "Y"),
        ("El Paso", "Santa Fe", 2, "X"),
        ("Santa Fe", "Oklahoma City", 3, "B"),
        ("Oklahoma City", "Denver", 4, "R"),
        ("Santa Fe", "Denver", 2, "X"),
        ("Denver", "Kansas City", 4, "K"),
        ("Denver", "Kansas City", 4, "O"),
        ("Denver", "Omaha", 4, "P"),
        ("New Orleans", "Miami", 6, "R"),
        ("New Orleans", "Atlanta", 4, "O"),
        ("New Orleans", "Atlanta", 4, "Y"),
        ("New Orleans", "Little Rock", 3, "G"),
        ("Little Rock", "Nashville", 3, "W"),
        ("Little Rock", "St. Louis", 2, "X"),
        ("St. Louis", "Nashville", 2, "X"),
        ("St. Louis", "Pittsburgh", 5, "G"),
        ("St. Louis", "Chicago", 2, "G"),
        ("St. Louis", "Chicago", 2, "W"),
        ("Chicago", "Pittsburgh", 3, "K"),
        ("Chicago", "Pittsburgh", 3, "O"),
        ("Chicago", "Toronto", 4, "W"),
        ("Sault St. Marie", "Montreal", 5, "K"),
        ("Toronto", "Montreal", 3, "X"),
        ("Sault St. Marie", "Toronto", 2, "X"),
        ("Toronto", "Pittsburgh", 2, "X"),
        ("Pittsburgh", "New York", 2, "W"),
        ("Pittsburgh", "New York", 2, "G"),
        ("Pittsburgh", "Washington", 2, "X"),
        ("Pittsburgh", "Raleigh", 2, "X"),
        ("Nashville", "Raleigh", 3, "K"),
        ("Nashville", "Atlanta", 1, "X"),
        ("Nashville", "Pittsburgh", 4, "Y"),
        ("Atlanta", "Miami", 5, "B"),
        ("Atlanta", "Charleston", 2, "X"),
        ("Atlanta", "Raleigh", 2, "X"),
        ("Atlanta", "Raleigh", 2, "X"),
        ("Charleston", "Miami", 4, "P"),
        ("Raleigh", "Charleston", 2, "X"),
        ("Raleigh", "Washington", 2, "X"),
        ("Raleigh", "Washington", 2, "X"),
        ("Washington", "New York", 2, "O"),
        ("Washington", "New York", 2, "K"),
        ("New York", "Boston", 2, "Y"),
        ("New York", "Boston", 2, "R"),
        ("New York", "Montreal", 3, "B"),
        ("Boston", "Montreal", 2, "X"),
        ("Boston", "Montreal", 2, "X"),
    ];

    let mut routes = [RouteDef {
        id: 0,
        city_a: 0,
        city_b: 0,
        length: 0,
        color: 255,
        double_pair_id: 255,
    }; USA_NUM_ROUTES];

    for i in 0..USA_NUM_ROUTES {
        let (ca_str, cb_str, len, col_str) = raw[i];
        let ca = city_name_to_index(ca_str).unwrap();
        let cb = city_name_to_index(cb_str).unwrap();
        let col = match col_str {
            "P" => 0,
            "W" => 1,
            "B" => 2,
            "Y" => 3,
            "O" => 4,
            "K" => 5,
            "R" => 6,
            "G" => 7,
            _ => 255,
        };
        routes[i] = RouteDef {
            id: i as u8,
            city_a: ca.min(cb),
            city_b: ca.max(cb),
            length: len,
            color: col,
            double_pair_id: 255,
        };
    }

    // Match double route pairs
    for i in 0..USA_NUM_ROUTES {
        for j in (i + 1)..USA_NUM_ROUTES {
            if routes[i].city_a == routes[j].city_a && routes[i].city_b == routes[j].city_b {
                routes[i].double_pair_id = j as u8;
                routes[j].double_pair_id = i as u8;
            }
        }
    }

    routes
}

pub fn get_usa_tickets() -> [TicketDef; USA_NUM_TICKETS] {
    let raw: [(&str, &str, u8); USA_NUM_TICKETS] = [
        ("Los Angeles", "New York", 21),
        ("Duluth", "Houston", 8),
        ("Sault St. Marie", "Nashville", 8),
        ("New York", "Atlanta", 6),
        ("Portland", "Nashville", 17),
        ("Vancouver", "Montreal", 20),
        ("Duluth", "El Paso", 10),
        ("Toronto", "Miami", 10),
        ("Portland", "Phoenix", 11),
        ("Dallas", "New York", 11),
        ("Calgary", "Salt Lake City", 7),
        ("Calgary", "Phoenix", 13),
        ("Los Angeles", "Miami", 20),
        ("Winnipeg", "Little Rock", 11),
        ("San Francisco", "Atlanta", 17),
        ("Kansas City", "Houston", 5),
        ("Los Angeles", "Chicago", 16),
        ("Denver", "Pittsburgh", 11),
        ("Chicago", "Santa Fe", 9),
        ("Vancouver", "Santa Fe", 13),
        ("Boston", "Miami", 12),
        ("Chicago", "New Orleans", 7),
        ("Montreal", "Atlanta", 9),
        ("Seattle", "New York", 22),
        ("Denver", "El Paso", 4),
        ("Helena", "Los Angeles", 8),
        ("Winnipeg", "Houston", 12),
        ("Montreal", "New Orleans", 13),
        ("Sault St. Marie", "Oklahoma City", 9),
        ("Seattle", "Los Angeles", 9),
    ];

    let mut tickets = [TicketDef {
        id: 0,
        city_a: 0,
        city_b: 0,
        points: 0,
    }; USA_NUM_TICKETS];

    for i in 0..USA_NUM_TICKETS {
        let (ca, cb, pts) = raw[i];
        tickets[i] = TicketDef {
            id: i as u8,
            city_a: city_name_to_index(ca).unwrap(),
            city_b: city_name_to_index(cb).unwrap(),
            points: pts,
        };
    }
    tickets
}

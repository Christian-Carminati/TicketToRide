"""Predefined maps and map loaders for Ticket to Ride."""

from src.game.board import Board, City
from src.game.card import CardColor
from src.game.route import Route
from src.game.ticket import DestinationTicket

# USA Cities: Name -> (X, Y)
USA_CITIES: dict[str, tuple[float, float]] = {
    "Atlanta": (0.85, 0.35),
    "Boston": (0.98, 0.75),
    "Calgary": (0.25, 0.95),
    "Charleston": (0.95, 0.35),
    "Chicago": (0.75, 0.65),
    "Dallas": (0.65, 0.1),
    "Denver": (0.5, 0.45),
    "Duluth": (0.65, 0.8),
    "El Paso": (0.55, 0.05),
    "Helena": (0.35, 0.75),
    "Houston": (0.7, 0.05),
    "Kansas City": (0.65, 0.4),
    "Las Vegas": (0.2, 0.15),
    "Little Rock": (0.7, 0.3),
    "Los Angeles": (0.1, 0.05),
    "Miami": (0.95, 0.05),
    "Montreal": (0.9, 0.85),
    "Nashville": (0.8, 0.45),
    "New Orleans": (0.8, 0.1),
    "New York": (0.95, 0.65),
    "Oklahoma City": (0.6, 0.25),
    "Omaha": (0.65, 0.55),
    "Phoenix": (0.35, 0.15),
    "Pittsburgh": (0.85, 0.6),
    "Portland": (0.08, 0.75),
    "Raleigh": (0.9, 0.45),
    "Salt Lake City": (0.3, 0.45),
    "San Francisco": (0.05, 0.35),
    "Santa Fe": (0.5, 0.25),
    "Sault St. Marie": (0.85, 0.85),
    "Seattle": (0.1, 0.85),
    "St. Louis": (0.7, 0.45),
    "Toronto": (0.8, 0.75),
    "Vancouver": (0.05, 0.95),
    "Washington": (0.9, 0.55),
    "Winnipeg": (0.45, 0.9),
}

# Raw USA Routes: (CityA, CityB, Length, ColorCode)
# ColorCode: P (Purple), W (White), B (Blue), Y (Yellow), O (Orange), K (Black), R (Red), G (Green), X (Gray)
USA_RAW_ROUTES: list[tuple[str, str, int, str]] = [
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
]

# USA Destination Tickets: (CityA, CityB, Points)
USA_RAW_TICKETS: list[tuple[str, str, int]] = [
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
]

COLOR_MAP: dict[str, CardColor | None] = {
    "P": CardColor.PURPLE,
    "W": CardColor.WHITE,
    "B": CardColor.BLUE,
    "Y": CardColor.YELLOW,
    "O": CardColor.ORANGE,
    "K": CardColor.BLACK,
    "R": CardColor.RED,
    "G": CardColor.GREEN,
    "X": None,  # Gray
}


def load_usa_board() -> tuple[Board, list[DestinationTicket]]:
    """Load the official Ticket to Ride USA board with 36 cities, 100 routes, and 30 tickets."""
    board = Board()
    for name, (x, y) in USA_CITIES.items():
        board.add_city(City(id=name.lower().replace(" ", "_").replace(".", ""), name=name, x=x, y=y))

    # Process routes and double routes
    routes: list[Route] = []
    seen_pairs: dict[tuple[str, str], list[int]] = {}

    for idx, (c_a, c_b, length, color_code) in enumerate(USA_RAW_ROUTES):
        route_id = f"r_{idx}_{c_a[:3].lower()}_{c_b[:3].lower()}"
        color = COLOR_MAP[color_code]
        r = Route(id=route_id, city_a=c_a, city_b=c_b, length=length, color=color)
        routes.append(r)

        pair_key = (min(c_a, c_b), max(c_a, c_b))
        if pair_key not in seen_pairs:
            seen_pairs[pair_key] = []
        seen_pairs[pair_key].append(idx)

    # Link double routes
    for indices in seen_pairs.values():
        if len(indices) == 2:
            r1 = routes[indices[0]]
            r2 = routes[indices[1]]
            r1.double_route_pair_id = r2.id
            r2.double_route_pair_id = r1.id

    board.routes = routes

    tickets: list[DestinationTicket] = []
    for idx, (c_a, c_b, points) in enumerate(USA_RAW_TICKETS):
        ticket_id = f"t_{idx}_{c_a[:3].lower()}_{c_b[:3].lower()}"
        tickets.append(DestinationTicket(id=ticket_id, city_a=c_a, city_b=c_b, points=points))

    return board, tickets


def create_synthetic_mini_board() -> tuple[Board, list[DestinationTicket]]:
    """Create a small 5-city synthetic board for fast unit and property tests."""
    board = Board()
    mini_cities = [
        City(id="A", name="City_A", x=0.0, y=0.0),
        City(id="B", name="City_B", x=1.0, y=0.0),
        City(id="C", name="City_C", x=2.0, y=1.0),
        City(id="D", name="City_D", x=1.0, y=2.0),
        City(id="E", name="City_E", x=0.0, y=2.0),
    ]
    for c in mini_cities:
        board.add_city(c)

    r1 = Route(id="r_ab_1", city_a="City_A", city_b="City_B", length=2, color=CardColor.RED)
    r2 = Route(id="r_ab_2", city_a="City_A", city_b="City_B", length=2, color=CardColor.BLUE)
    r1.double_route_pair_id = r2.id
    r2.double_route_pair_id = r1.id

    r3 = Route(id="r_bc", city_a="City_B", city_b="City_C", length=3, color=CardColor.GREEN)
    r4 = Route(id="r_cd", city_a="City_C", city_b="City_D", length=2, color=None)
    r5 = Route(id="r_de", city_a="City_D", city_b="City_E", length=2, color=CardColor.YELLOW)
    r6 = Route(id="r_ea", city_a="City_E", city_b="City_A", length=4, color=CardColor.BLACK)

    board.routes = [r1, r2, r3, r4, r5, r6]

    tickets = [
        DestinationTicket(id="t_ac", city_a="City_A", city_b="City_C", points=5),
        DestinationTicket(id="t_ae", city_a="City_A", city_b="City_E", points=4),
        DestinationTicket(id="t_bd", city_a="City_B", city_b="City_D", points=5),
        DestinationTicket(id="t_ce", city_a="City_C", city_b="City_E", points=6),
        DestinationTicket(id="t_ad", city_a="City_A", city_b="City_D", points=7),
        DestinationTicket(id="t_be", city_a="City_B", city_b="City_E", points=6),
        DestinationTicket(id="t_ab", city_a="City_A", city_b="City_B", points=2),
    ]

    return board, tickets


# Europe Cities: Name -> (X, Y)
EUROPE_CITIES: dict[str, tuple[float, float]] = {
    "Amsterdam": (0.35, 0.65),
    "Angora": (0.85, 0.20),
    "Athina": (0.75, 0.15),
    "Barcelona": (0.18, 0.32),
    "Berlin": (0.50, 0.68),
    "Brest": (0.15, 0.60),
    "Brindisi": (0.62, 0.25),
    "Bruxelles": (0.32, 0.60),
    "Bucuresti": (0.78, 0.35),
    "Budapest": (0.62, 0.45),
    "Cadiz": (0.05, 0.15),
    "Constantinople": (0.82, 0.25),
    "Danzig": (0.58, 0.72),
    "Dieppe": (0.25, 0.62),
    "Edinburgh": (0.22, 0.85),
    "Erzurum": (0.95, 0.25),
    "Essen": (0.40, 0.64),
    "Frankfurt": (0.42, 0.58),
    "Kharkov": (0.90, 0.60),
    "Kobenhavn": (0.48, 0.80),
    "Kyiv": (0.78, 0.58),
    "Lisboa": (0.02, 0.22),
    "London": (0.24, 0.70),
    "Madrid": (0.10, 0.28),
    "Marseille": (0.32, 0.40),
    "Moskva": (0.88, 0.75),
    "Munchen": (0.46, 0.50),
    "Palermo": (0.55, 0.15),
    "Paris": (0.28, 0.54),
    "Petrograd": (0.82, 0.88),
    "Riga": (0.68, 0.78),
    "Roma": (0.48, 0.30),
    "Rostov": (0.95, 0.50),
    "Sarajevo": (0.60, 0.35),
    "Sevastopol": (0.88, 0.42),
    "Smolensk": (0.82, 0.70),
    "Smyrna": (0.80, 0.15),
    "Sofia": (0.70, 0.30),
    "Stockholm": (0.58, 0.88),
    "Venezia": (0.48, 0.42),
    "Warszawa": (0.65, 0.65),
    "Wien": (0.55, 0.50),
    "Wilno": (0.72, 0.70),
    "Zagreb": (0.54, 0.40),
    "Zurich": (0.38, 0.48),
}

# Raw Europe Routes: (CityA, CityB, Length, ColorCode)
EUROPE_RAW_ROUTES: list[tuple[str, str, int, str]] = [
    ("Edinburgh", "London", 4, "K"),
    ("Edinburgh", "London", 4, "O"),
    ("London", "Dieppe", 2, "X"),
    ("London", "Dieppe", 2, "X"),
    ("London", "Amsterdam", 2, "X"),
    ("Brest", "Dieppe", 2, "X"),
    ("Brest", "Paris", 3, "K"),
    ("Brest", "Madrid", 4, "P"),
    ("Dieppe", "Paris", 1, "P"),
    ("Dieppe", "Bruxelles", 2, "G"),
    ("Amsterdam", "Bruxelles", 1, "X"),
    ("Amsterdam", "Essen", 3, "Y"),
    ("Amsterdam", "Frankfurt", 2, "W"),
    ("Bruxelles", "Paris", 2, "Y"),
    ("Bruxelles", "Paris", 2, "R"),
    ("Bruxelles", "Frankfurt", 2, "B"),
    ("Paris", "Frankfurt", 3, "W"),
    ("Paris", "Frankfurt", 3, "O"),
    ("Paris", "Zurich", 3, "X"),
    ("Paris", "Marseille", 4, "X"),
    ("Marseille", "Zurich", 2, "P"),
    ("Marseille", "Roma", 4, "X"),
    ("Marseille", "Barcelona", 4, "X"),
    ("Lisboa", "Madrid", 3, "P"),
    ("Lisboa", "Cadiz", 2, "B"),
    ("Cadiz", "Madrid", 3, "O"),
    ("Madrid", "Barcelona", 2, "Y"),
    ("Frankfurt", "Essen", 2, "G"),
    ("Frankfurt", "Berlin", 3, "K"),
    ("Frankfurt", "Berlin", 3, "R"),
    ("Frankfurt", "Munchen", 2, "P"),
    ("Essen", "Berlin", 2, "B"),
    ("Essen", "Kobenhavn", 3, "X"),
    ("Essen", "Kobenhavn", 3, "X"),
    ("Kobenhavn", "Stockholm", 3, "Y"),
    ("Kobenhavn", "Stockholm", 3, "W"),
    ("Stockholm", "Petrograd", 8, "X"),
    ("Berlin", "Danzig", 4, "X"),
    ("Berlin", "Warszawa", 4, "P"),
    ("Berlin", "Warszawa", 4, "Y"),
    ("Berlin", "Wien", 3, "G"),
    ("Munchen", "Zurich", 2, "Y"),
    ("Munchen", "Wien", 3, "O"),
    ("Munchen", "Venezia", 2, "B"),
    ("Zurich", "Venezia", 2, "G"),
    ("Venezia", "Roma", 2, "K"),
    ("Venezia", "Zagreb", 2, "X"),
    ("Roma", "Palermo", 4, "X"),
    ("Roma", "Brindisi", 2, "W"),
    ("Palermo", "Brindisi", 3, "X"),
    ("Palermo", "Smyrna", 6, "X"),
    ("Brindisi", "Athina", 4, "X"),
    ("Zagreb", "Wien", 2, "X"),
    ("Zagreb", "Budapest", 2, "O"),
    ("Zagreb", "Sarajevo", 3, "R"),
    ("Sarajevo", "Budapest", 3, "P"),
    ("Sarajevo", "Athina", 4, "G"),
    ("Sarajevo", "Sofia", 2, "X"),
    ("Athina", "Sofia", 3, "P"),
    ("Athina", "Smyrna", 2, "X"),
    ("Sofia", "Bucuresti", 2, "X"),
    ("Sofia", "Constantinople", 3, "B"),
    ("Constantinople", "Smyrna", 2, "Y"),
    ("Constantinople", "Bucuresti", 3, "Y"),
    ("Constantinople", "Angora", 2, "X"),
    ("Smyrna", "Angora", 3, "O"),
    ("Angora", "Erzurum", 3, "K"),
    ("Wien", "Budapest", 1, "R"),
    ("Wien", "Budapest", 1, "W"),
    ("Wien", "Warszawa", 4, "B"),
    ("Budapest", "Bucuresti", 4, "X"),
    ("Budapest", "Kyiv", 6, "X"),
    ("Bucuresti", "Sevastopol", 4, "W"),
    ("Bucuresti", "Kyiv", 4, "X"),
    ("Danzig", "Warszawa", 2, "X"),
    ("Danzig", "Riga", 3, "K"),
    ("Warszawa", "Wilno", 3, "R"),
    ("Warszawa", "Kyiv", 4, "X"),
    ("Riga", "Petrograd", 4, "X"),
    ("Riga", "Wilno", 4, "G"),
    ("Riga", "Smolensk", 3, "X"),
    ("Wilno", "Petrograd", 4, "B"),
    ("Wilno", "Smolensk", 3, "Y"),
    ("Wilno", "Kyiv", 2, "X"),
    ("Kyiv", "Smolensk", 3, "R"),
    ("Kyiv", "Kharkov", 4, "X"),
    ("Smolensk", "Moskva", 2, "O"),
    ("Petrograd", "Moskva", 4, "W"),
    ("Moskva", "Kharkov", 4, "P"),
    ("Kharkov", "Rostov", 2, "G"),
    ("Rostov", "Sevastopol", 4, "X"),
    ("Rostov", "Erzurum", 5, "X"),
    ("Sevastopol", "Erzurum", 4, "X"),
    ("Sevastopol", "Constantinople", 4, "X"),
]

# Raw Europe Destination Tickets: (CityA, CityB, Points)
EUROPE_RAW_TICKETS: list[tuple[str, str, int]] = [
    # Long tickets
    ("Brest", "Petrograd", 20),
    ("Cadiz", "Stockholm", 21),
    ("Edinburgh", "Athina", 21),
    ("Kobenhavn", "Erzurum", 21),
    ("Lisboa", "Danzig", 20),
    ("Palermo", "Moskva", 20),
    # Regular tickets
    ("Amsterdam", "Madrid", 12),
    ("Amsterdam", "Roma", 8),
    ("Athina", "Angora", 5),
    ("Angora", "Kharkov", 10),
    ("Barcelona", "Bruxelles", 8),
    ("Barcelona", "Munchen", 8),
    ("Berlin", "Bucuresti", 8),
    ("Berlin", "Moskva", 12),
    ("Berlin", "Roma", 9),
    ("Brest", "Marseille", 7),
    ("Brest", "Venezia", 8),
    ("Bruxelles", "Danzig", 9),
    ("Budapest", "Sofia", 5),
    ("Dieppe", "Marseille", 8),
    ("Edinburgh", "Paris", 7),
    ("Essen", "Kyiv", 10),
    ("Frankfurt", "Kobenhavn", 5),
    ("Frankfurt", "Smolensk", 13),
    ("London", "Wien", 10),
    ("London", "Berlin", 7),
    ("Madrid", "Dieppe", 8),
    ("Marseille", "Essen", 8),
    ("Paris", "Wien", 8),
    ("Paris", "Zagreb", 7),
    ("Petrograd", "Kyiv", 8),
    ("Riga", "Bucuresti", 10),
    ("Roma", "Smyrna", 8),
    ("Rostov", "Erzurum", 5),
    ("Sarajevo", "Sevastopol", 8),
    ("Smolensk", "Rostov", 8),
    ("Sofia", "Smyrna", 5),
    ("Stockholm", "Wien", 11),
    ("Venezia", "Constantinople", 10),
    ("Warszawa", "Smolensk", 6),
    ("Zagreb", "Brindisi", 6),
    ("Zurich", "Brindisi", 6),
    ("Zurich", "Budapest", 6),
]


def load_europe_board() -> tuple[Board, list[DestinationTicket]]:
    """Load the official Ticket to Ride Europe board."""
    board = Board()
    for name, (x, y) in EUROPE_CITIES.items():
        board.add_city(City(id=name.lower().replace(" ", "_"), name=name, x=x, y=y))

    routes: list[Route] = []
    seen_pairs: dict[tuple[str, str], list[int]] = {}

    for idx, (c_a, c_b, length, color_code) in enumerate(EUROPE_RAW_ROUTES):
        route_id = f"eur_r_{idx}_{c_a[:3].lower()}_{c_b[:3].lower()}"
        color = COLOR_MAP[color_code]
        r = Route(id=route_id, city_a=c_a, city_b=c_b, length=length, color=color)
        routes.append(r)

        pair_key = (min(c_a, c_b), max(c_a, c_b))
        if pair_key not in seen_pairs:
            seen_pairs[pair_key] = []
        seen_pairs[pair_key].append(idx)

    for indices in seen_pairs.values():
        if len(indices) == 2:
            r1 = routes[indices[0]]
            r2 = routes[indices[1]]
            r1.double_route_pair_id = r2.id
            r2.double_route_pair_id = r1.id

    board.routes = routes
    board.rebuild_indexes()

    tickets: list[DestinationTicket] = []
    for idx, (c_a, c_b, points) in enumerate(EUROPE_RAW_TICKETS):
        ticket_id = f"eur_t_{idx}_{c_a[:3].lower()}_{c_b[:3].lower()}"
        tickets.append(DestinationTicket(id=ticket_id, city_a=c_a, city_b=c_b, points=points))

    return board, tickets


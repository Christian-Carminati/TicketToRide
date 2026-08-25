/**
 * Map coordinates and route configurations for USA and Mini maps.
 */

export interface BoardCity {
  id: string;
  name: string;
  x: number; // 0..1
  y: number; // 0..1
}

export interface BoardRoute {
  id: string;
  city_a: string;
  city_b: string;
  length: number;
  color: string | null; // e.g. "RED", "BLUE", "GREEN", "YELLOW", "ORANGE", "PURPLE", "BLACK", "WHITE", or null (Gray)
  double_pair_id?: string;
  offset_index?: number; // 0 for single, -1 or 1 for parallel double routes
}

export const COLOR_HEX: Record<string, string> = {
  PURPLE: '#7E22CE',
  WHITE: '#F5F0E6',
  BLUE: '#1D4ED8',
  YELLOW: '#D97706',
  ORANGE: '#C2410C',
  BLACK: '#1E1B18',
  RED: '#B91C1C',
  GREEN: '#15803D',
  LOCOMOTIVE: '#E11D48',
  GRAY: '#7D6A5A',
};

export const USA_CITIES: BoardCity[] = [
  { id: 'atlanta', name: 'Atlanta', x: 0.85, y: 0.35 },
  { id: 'boston', name: 'Boston', x: 0.98, y: 0.75 },
  { id: 'calgary', name: 'Calgary', x: 0.25, y: 0.95 },
  { id: 'charleston', name: 'Charleston', x: 0.95, y: 0.35 },
  { id: 'chicago', name: 'Chicago', x: 0.75, y: 0.65 },
  { id: 'dallas', name: 'Dallas', x: 0.65, y: 0.1 },
  { id: 'denver', name: 'Denver', x: 0.5, y: 0.45 },
  { id: 'duluth', name: 'Duluth', x: 0.65, y: 0.8 },
  { id: 'el_paso', name: 'El Paso', x: 0.55, y: 0.05 },
  { id: 'helena', name: 'Helena', x: 0.35, y: 0.75 },
  { id: 'houston', name: 'Houston', x: 0.7, y: 0.05 },
  { id: 'kansas_city', name: 'Kansas City', x: 0.65, y: 0.4 },
  { id: 'las_vegas', name: 'Las Vegas', x: 0.2, y: 0.15 },
  { id: 'little_rock', name: 'Little Rock', x: 0.7, y: 0.3 },
  { id: 'los_angeles', name: 'Los Angeles', x: 0.1, y: 0.05 },
  { id: 'miami', name: 'Miami', x: 0.95, y: 0.05 },
  { id: 'montreal', name: 'Montreal', x: 0.9, y: 0.85 },
  { id: 'nashville', name: 'Nashville', x: 0.8, y: 0.45 },
  { id: 'new_orleans', name: 'New Orleans', x: 0.8, y: 0.1 },
  { id: 'new_york', name: 'New York', x: 0.95, y: 0.65 },
  { id: 'oklahoma_city', name: 'Oklahoma City', x: 0.6, y: 0.25 },
  { id: 'omaha', name: 'Omaha', x: 0.65, y: 0.55 },
  { id: 'phoenix', name: 'Phoenix', x: 0.35, y: 0.15 },
  { id: 'pittsburgh', name: 'Pittsburgh', x: 0.85, y: 0.6 },
  { id: 'portland', name: 'Portland', x: 0.08, y: 0.75 },
  { id: 'raleigh', name: 'Raleigh', x: 0.9, y: 0.45 },
  { id: 'salt_lake_city', name: 'Salt Lake City', x: 0.3, y: 0.45 },
  { id: 'san_francisco', name: 'San Francisco', x: 0.05, y: 0.35 },
  { id: 'santa_fe', name: 'Santa Fe', x: 0.5, y: 0.25 },
  { id: 'sault_st_marie', name: 'Sault St. Marie', x: 0.85, y: 0.85 },
  { id: 'seattle', name: 'Seattle', x: 0.1, y: 0.85 },
  { id: 'st_louis', name: 'St. Louis', x: 0.7, y: 0.45 },
  { id: 'toronto', name: 'Toronto', x: 0.8, y: 0.75 },
  { id: 'vancouver', name: 'Vancouver', x: 0.05, y: 0.95 },
  { id: 'washington', name: 'Washington', x: 0.9, y: 0.55 },
  { id: 'winnipeg', name: 'Winnipeg', x: 0.45, y: 0.9 },
];

export const USA_RAW_ROUTES_DATA: Array<[string, string, number, string | null]> = [
  ['Vancouver', 'Calgary', 3, null],
  ['Vancouver', 'Seattle', 1, null],
  ['Vancouver', 'Seattle', 1, null],
  ['Seattle', 'Calgary', 4, null],
  ['Seattle', 'Helena', 6, 'YELLOW'],
  ['Seattle', 'Portland', 1, null],
  ['Seattle', 'Portland', 1, null],
  ['Portland', 'Salt Lake City', 6, 'BLUE'],
  ['Portland', 'San Francisco', 5, 'GREEN'],
  ['Portland', 'San Francisco', 5, 'PURPLE'],
  ['San Francisco', 'Salt Lake City', 5, 'ORANGE'],
  ['San Francisco', 'Salt Lake City', 5, 'WHITE'],
  ['San Francisco', 'Los Angeles', 3, 'YELLOW'],
  ['San Francisco', 'Los Angeles', 3, 'PURPLE'],
  ['Los Angeles', 'Las Vegas', 2, null],
  ['Los Angeles', 'Phoenix', 3, null],
  ['Los Angeles', 'El Paso', 6, 'BLACK'],
  ['Calgary', 'Winnipeg', 6, 'WHITE'],
  ['Calgary', 'Helena', 4, null],
  ['Helena', 'Winnipeg', 4, 'BLUE'],
  ['Helena', 'Salt Lake City', 3, 'PURPLE'],
  ['Helena', 'Denver', 4, 'GREEN'],
  ['Helena', 'Duluth', 6, 'ORANGE'],
  ['Helena', 'Omaha', 5, 'RED'],
  ['Salt Lake City', 'Denver', 3, 'RED'],
  ['Salt Lake City', 'Denver', 3, 'YELLOW'],
  ['Las Vegas', 'Salt Lake City', 3, 'ORANGE'],
  ['Phoenix', 'Denver', 5, 'WHITE'],
  ['Phoenix', 'Santa Fe', 3, null],
  ['Phoenix', 'El Paso', 3, null],
  ['Winnipeg', 'Sault St. Marie', 6, null],
  ['Winnipeg', 'Duluth', 4, 'BLACK'],
  ['Duluth', 'Sault St. Marie', 3, null],
  ['Duluth', 'Toronto', 6, 'PURPLE'],
  ['Duluth', 'Chicago', 3, 'RED'],
  ['Duluth', 'Omaha', 2, null],
  ['Duluth', 'Omaha', 2, null],
  ['Omaha', 'Chicago', 4, 'BLUE'],
  ['Omaha', 'Kansas City', 1, null],
  ['Omaha', 'Kansas City', 1, null],
  ['Kansas City', 'St. Louis', 2, 'BLUE'],
  ['Kansas City', 'St. Louis', 2, 'PURPLE'],
  ['Kansas City', 'Oklahoma City', 2, null],
  ['Kansas City', 'Oklahoma City', 2, null],
  ['Oklahoma City', 'Little Rock', 2, null],
  ['Oklahoma City', 'Dallas', 2, null],
  ['Oklahoma City', 'Dallas', 2, null],
  ['Dallas', 'Little Rock', 2, null],
  ['Dallas', 'Houston', 1, null],
  ['Dallas', 'Houston', 1, null],
  ['Houston', 'New Orleans', 2, null],
  ['El Paso', 'Houston', 6, 'GREEN'],
  ['El Paso', 'Dallas', 4, 'RED'],
  ['El Paso', 'Oklahoma City', 5, 'YELLOW'],
  ['El Paso', 'Santa Fe', 2, null],
  ['Santa Fe', 'Oklahoma City', 3, 'BLUE'],
  ['Oklahoma City', 'Denver', 4, 'RED'],
  ['Santa Fe', 'Denver', 2, null],
  ['Denver', 'Kansas City', 4, 'BLACK'],
  ['Denver', 'Kansas City', 4, 'ORANGE'],
  ['Denver', 'Omaha', 4, 'PURPLE'],
  ['New Orleans', 'Miami', 6, 'RED'],
  ['New Orleans', 'Atlanta', 4, 'ORANGE'],
  ['New Orleans', 'Atlanta', 4, 'YELLOW'],
  ['New Orleans', 'Little Rock', 3, 'GREEN'],
  ['Little Rock', 'Nashville', 3, 'WHITE'],
  ['Little Rock', 'St. Louis', 2, null],
  ['St. Louis', 'Nashville', 2, null],
  ['St. Louis', 'Pittsburgh', 5, 'GREEN'],
  ['St. Louis', 'Chicago', 2, 'GREEN'],
  ['St. Louis', 'Chicago', 2, 'WHITE'],
  ['Chicago', 'Pittsburgh', 3, 'BLACK'],
  ['Chicago', 'Pittsburgh', 3, 'ORANGE'],
  ['Chicago', 'Toronto', 4, 'WHITE'],
  ['Sault St. Marie', 'Montreal', 5, 'BLACK'],
  ['Toronto', 'Montreal', 3, null],
  ['Sault St. Marie', 'Toronto', 2, null],
  ['Toronto', 'Pittsburgh', 2, null],
  ['Pittsburgh', 'New York', 2, 'WHITE'],
  ['Pittsburgh', 'New York', 2, 'GREEN'],
  ['Pittsburgh', 'Washington', 2, null],
  ['Pittsburgh', 'Raleigh', 2, null],
  ['Nashville', 'Raleigh', 3, 'BLACK'],
  ['Nashville', 'Atlanta', 1, null],
  ['Nashville', 'Pittsburgh', 4, 'YELLOW'],
  ['Atlanta', 'Miami', 5, 'BLUE'],
  ['Atlanta', 'Charleston', 2, null],
  ['Atlanta', 'Raleigh', 2, null],
  ['Atlanta', 'Raleigh', 2, null],
  ['Charleston', 'Miami', 4, 'PURPLE'],
  ['Raleigh', 'Charleston', 2, null],
  ['Raleigh', 'Washington', 2, null],
  ['Raleigh', 'Washington', 2, null],
  ['Washington', 'New York', 2, 'ORANGE'],
  ['Washington', 'New York', 2, 'BLACK'],
  ['New York', 'Boston', 2, 'YELLOW'],
  ['New York', 'Boston', 2, 'RED'],
  ['New York', 'Montreal', 3, 'BLUE'],
  ['Boston', 'Montreal', 2, null],
  ['Boston', 'Montreal', 2, null],
];

export const MINI_CITIES: BoardCity[] = [
  { id: 'A', name: 'City_A', x: 0.15, y: 0.2 },
  { id: 'B', name: 'City_B', x: 0.85, y: 0.2 },
  { id: 'C', name: 'City_C', x: 0.85, y: 0.8 },
  { id: 'D', name: 'City_D', x: 0.5, y: 0.9 },
  { id: 'E', name: 'City_E', x: 0.15, y: 0.8 },
];

export const MINI_ROUTES: BoardRoute[] = [
  { id: 'r_ab_1', city_a: 'City_A', city_b: 'City_B', length: 2, color: 'RED', offset_index: -1 },
  { id: 'r_ab_2', city_a: 'City_A', city_b: 'City_B', length: 2, color: 'BLUE', offset_index: 1 },
  { id: 'r_bc', city_a: 'City_B', city_b: 'City_C', length: 3, color: 'GREEN', offset_index: 0 },
  { id: 'r_cd', city_a: 'City_C', city_b: 'City_D', length: 2, color: null, offset_index: 0 },
  { id: 'r_de', city_a: 'City_D', city_b: 'City_E', length: 2, color: 'YELLOW', offset_index: 0 },
  { id: 'r_ea', city_a: 'City_E', city_b: 'City_A', length: 4, color: 'BLACK', offset_index: 0 },
];

export function buildUsaRoutes(): BoardRoute[] {
  const routes: BoardRoute[] = [];
  const pairCounts: Record<string, number> = {};

  USA_RAW_ROUTES_DATA.forEach(([cA, cB, length, color], idx) => {
    const pairKey = [cA, cB].sort().join('::');
    const occurrence = (pairCounts[pairKey] || 0) + 1;
    pairCounts[pairKey] = occurrence;

    const routeId = `r_${idx}_${cA.slice(0, 3).toLowerCase()}_${cB.slice(0, 3).toLowerCase()}`;
    routes.push({
      id: routeId,
      city_a: cA,
      city_b: cB,
      length,
      color,
      offset_index: occurrence === 2 ? 1 : 0,
    });
  });

  // Assign offset_index -1 to first route of pair if pair has 2 routes
  routes.forEach((r) => {
    const pairKey = [r.city_a, r.city_b].sort().join('::');
    if (pairCounts[pairKey] === 2 && r.offset_index === 0) {
      r.offset_index = -1;
    }
  });

  return routes;
}

export function getMapData(mapName: string = 'usa'): { cities: BoardCity[]; routes: BoardRoute[] } {
  const normalized = mapName?.toLowerCase() || 'usa';
  if (normalized === 'mini') {
    return {
      cities: MINI_CITIES,
      routes: MINI_ROUTES,
    };
  }
  return {
    cities: USA_CITIES,
    routes: buildUsaRoutes(),
  };
}

export const BOARD_CITIES = USA_CITIES;
export const buildBoardRoutes = buildUsaRoutes;

# Steampunk UI & USA Vector Cartography Redesign — Technical Specification

## 1. Overview & Objectives

This specification defines the radical UI transformation of **TicketToRide RL Lab** into an authentic **Victorian Steampunk Cartographer & Analytical Laboratory** aesthetic.

### Key Objectives
1. **Parchment & Brass Design System**: Transition the entire web client from dark-slate cyberpunk to an immersive aged parchment substrate (`#F6EEDF` base, sepia inks `#23140C`, brass/copper/bronze bezels, rivet corner details, and Victorian typography).
2. **High-Clarity USA Vector Cartography**: Render a crisp, rich, geographic vector background map for North America and USA (coastlines, Great Lakes, hatched mountain topography, compass rose, latitude/longitude graticule lines, and ornamental survey cartouche) that makes geography visually clear behind the rail network.
3. **Victorian Analytical Instruments for RL Brain**: Transform neural inspection components into steam-powered laboratory instruments (brass steam pressure gauge for Critic Value Head, pneumatic frequency levers for action probabilities, thermionic vacuum tube filaments for tensor activations, and telegraph ticker tape for action logs).
4. **Period-Authentic Ephemera**: Style train cards as 19th-century railway shares/tickets, destination tickets as urgent Western Union telegrams with wax seals, and decks as spring-loaded brass card dispensers.
5. **Zero-Lag & Zero Functional Degradation**: Retain all interactive features, WebSocket 60Hz telemetry, frame-by-frame counterfactual scrubber, and partial observability modes without performance regression.

---

## 2. Visual Architecture & Design Tokens

### 2.1 Color Palette & Materials
- **Parchment Canvas Substrate**:
  - Base: `#F6EEDF` (light aged vellum) to `#EADABF` (warm center)
  - Edge Burn/Vignette: `#D8C3A0` to `#C5AA82`
  - Inset Shadow: `inset 0 0 40px rgba(110, 70, 30, 0.15)`
- **Inks & Typography**:
  - Primary Ink: `#23140C` (rich walnut brown / deep sepia)
  - Secondary Ink: `#4D311E` (aged brown)
  - Muted Ink: `#7A5B42` (engraver's wash)
  - Accent Gold/Brass: `#9E6B00` and `#B8860B`
- **Metals & Materials**:
  - Polished Brass: `linear-gradient(135deg, #E6C665 0%, #B8860B 50%, #8C6200 100%)`
  - Burnished Copper: `linear-gradient(135deg, #E29578 0%, #CD7F32 50%, #8D4925 100%)`
  - Cast Iron / Rivet: `#382E2B`
  - Wood Trim (Walnut): `linear-gradient(180deg, #4A2E1B 0%, #2E1B0F 100%)`
- **Steampunk Jewel Colors for Train Routes**:
  - Red: `#B91C1C` (Crimson ruby)
  - Blue: `#1D4ED8` (Sapphire)
  - Green: `#15803D` (Emerald)
  - Yellow: `#D97706` (Amber gold)
  - Orange: `#C2410C` (Terracotta copper)
  - Purple: `#7E22CE` (Amethyst)
  - Black: `#1E1B18` (Anthracite coal)
  - White: `#F5F0E6` (Pearl ivory)
  - Locomotive: `#E11D48` (Gilded rose locomotive)
  - Gray: `#8C7A6B` (Weathered stone)

### 2.2 Typography Hierarchy
- **Display & Headings**: `Cinzel Decorative`, `Playfair Display`, serif with engraved capitals.
- **UI & Panel Labels**: `Crimson Pro`, `Alegreya`, serif with high readability.
- **Data & Telemetry**: `JetBrains Mono`, `Courier Prime`, monospace typewriter format for neural tensors, logits, step counters, and timestamps.

---

## 3. High-Clarity USA Vector Cartography Engine

### 3.1 Geographic Landmass Layers in `BoardSVG.tsx`
The board SVG will render a calibrated multi-layer cartographic survey:
1. **Paper Background Texture & Graticule**:
   - Subtle SVG noise filter (`feTurbulence` + `feColorMatrix`) for aged fiber texture.
   - Longitude and latitude grid lines ($30^\circ\text{N}, 35^\circ\text{N}, 40^\circ\text{N}, 45^\circ\text{N}, 50^\circ\text{N}$ and $70^\circ\text{W}, 80^\circ\text{W}, 90^\circ\text{W}, 100^\circ\text{W}, 110^\circ\text{W}, 120^\circ\text{W}$) with subtle sepia labels.
2. **Coastline & Continent Contours (North America)**:
   - Pacific Coast from Vancouver to Baja California.
   - Atlantic Coast from Maine/Boston down through New York, Washington, Charleston, and Florida Peninsula.
   - Gulf of Mexico arc from Florida, New Orleans, Houston to Mexican coast.
   - Canadian territorial boundary line running east-west across northern US border.
3. **The Great Lakes & Major Waterways**:
   - Distinct, anatomically faithful outlines for Lake Superior, Lake Michigan, Lake Huron, Lake Erie, and Lake Ontario.
   - Filled with soft antique cyan/sepia water tint (`#D5E3E4` with engraved shoreline ripple lines).
4. **Mountain Ranges Relief (Hachure Engraving)**:
   - Stylized 19th-century hachure mountain peaks for the Rocky Mountains (spanning Helena, Salt Lake City, Denver, Santa Fe) and Appalachian Ridge (Pittsburgh, Nashville, Raleigh).
5. **Nautical Compass Rose & Cartouche**:
   - Elaborate multi-pointed brass compass star in the Atlantic quadrant.
   - Vintage scroll cartouche: *"UNITED STATES RAILWAY SURVEY — 1885"*.

### 3.2 City Nodes & Train Routes
- **City Nodes (`CityNode.tsx`)**:
  - Antiqued brass disc with center rivet and outer etched ring.
  - Hover / Highlight: Glowing brass mechanism ring with rotating dash pattern.
  - Text label: Engraved sepia text with ivory stroke backdrop for 100% legibility over land or water.
- **Route Edges (`RouteEdge.tsx`)**:
  - Dual-rail track bed with wooden ties.
  - High-saturation jewel colored train blocks.
  - Claimable state: Steam particle animation or pulsating brass border.

---

## 4. Analytical Instruments (Brain Inspector & Controls)

### 4.1 Steam Manometer Value Head (`ValueHeadGauge.tsx`)
- Round brass bezel with glass reflection arc.
- Curved gauge dial from $-1.0$ (Critical / Loss) to $+1.0$ (High Pressure / Win) or $0–100\%$.
- Damped swinging brass needle with center pivot rivet and PSI / value readout.

### 4.2 Action Probabilities & Masking (`ActionProbabilitiesChart.tsx`)
- Styled as Victorian pneumatic levers / brass frequency sliders.
- Valid actions displayed as warm amber/brass bars.
- Masked actions displayed with iron barred hash marks and "LOCKED" indicator.

### 4.3 Observation Tensor & Layer Activations (`ObservationTensorViewer.tsx`)
- Heatmap displayed as glowing thermionic Nixie / filament matrix with amber and copper warmth.

### 4.4 Scrubber Transport & Header (`ScrubberTransportBar.tsx`, `StudioHeader.tsx`)
- Master Header: Antiqued brass steam locomotive instrument plate with gear-toggle buttons and research cockpit status.
- Scrubber: Train conductor's chronometer with brass throttle slider and machined toggle switches.

---

## 5. Testing & Verification

1. **Visual Regression & Layout Checks**:
   - Validate interactive game, replay scrubber, live training, tournament arena, and reports views across desktop and laptop viewport sizes.
   - Ensure high contrast ($>4.5:1$) for all text on parchment backgrounds.
2. **Functionality Continuity**:
   - Verify claim route clicking, card drawing, bot step execution, and WebSocket 60Hz telemetry work smoothly without lag or frame drops.
3. **Automated Mechanical Checks**:
   - Run TypeScript type checking (`tsc --noEmit`) and Vite build (`npm run build` in `frontend/`).
   - Run Impeccable detector script (`node .../detect.mjs`) on modified files.

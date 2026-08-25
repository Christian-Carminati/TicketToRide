import React from 'react';

interface UsaCartographyBackgroundProps {
  width?: number;
  height?: number;
}

export const UsaCartographyBackground: React.FC<UsaCartographyBackgroundProps> = ({
  width = 1000,
  height = 650,
}) => {
  return (
    <g className="usa-cartography-background" style={{ pointerEvents: 'none', userSelect: 'none' }}>
      <defs>
        {/* Procedural parchment noise texture */}
        <filter id="carto-parchment-grain" x="0%" y="0%" width="100%" height="100%">
          <feTurbulence type="fractalNoise" baseFrequency="0.04" numOctaves="4" result="noise" />
          <feColorMatrix
            type="matrix"
            values="0 0 0 0 0.94   0 0 0 0 0.88   0 0 0 0 0.76   0 0 0 0.08 0"
            result="coloredNoise"
          />
          <feBlend mode="multiply" in="SourceGraphic" in2="coloredNoise" />
        </filter>

        {/* Ocean Waves / Coastal Engraving Pattern */}
        <pattern id="coastal-waves" width="24" height="12" patternUnits="userSpaceOnUse">
          <path
            d="M 0 6 Q 6 3 12 6 T 24 6"
            fill="none"
            stroke="rgba(80, 115, 125, 0.18)"
            strokeWidth="0.8"
          />
        </pattern>

        {/* Topography Hatch Pattern for Mountain Ridges */}
        <pattern id="mountain-hatch" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
          <line x1="0" y1="0" x2="0" y2="8" stroke="rgba(110, 70, 30, 0.22)" strokeWidth="0.8" />
        </pattern>

        {/* Gradients */}
        <radialGradient id="land-gradient" cx="52%" cy="46%" r="58%">
          <stop offset="0%" stopColor="#F9F3E5" />
          <stop offset="65%" stopColor="#F2E4CB" />
          <stop offset="100%" stopColor="#E2CFAC" />
        </radialGradient>

        <linearGradient id="ocean-gradient" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#D9E7E7" />
          <stop offset="50%" stopColor="#CDE0E0" />
          <stop offset="100%" stopColor="#C1D6D7" />
        </linearGradient>

        <linearGradient id="brass-frame-gradient" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#F5D77F" />
          <stop offset="35%" stopColor="#C89D2B" />
          <stop offset="70%" stopColor="#916707" />
          <stop offset="100%" stopColor="#5E3F02" />
        </linearGradient>

        <filter id="carto-shadow" x="-5%" y="-5%" width="110%" height="110%">
          <feDropShadow dx="2" dy="4" stdDeviation="5" floodColor="#3B2618" floodOpacity="0.25" />
        </filter>
      </defs>

      {/* 1. Base Ocean & Sea Water Foundation */}
      <rect width={width} height={height} fill="url(#ocean-gradient)" />
      <rect width={width} height={height} fill="url(#coastal-waves)" />

      {/* 2. Geodetic Graticule Lines (Latitude & Longitude Grid) */}
      <g className="graticule-layer" stroke="rgba(90, 60, 35, 0.14)" strokeWidth="0.75" strokeDasharray="4 4">
        {/* Latitudes */}
        <line x1={30} y1={85} x2={970} y2={85} />
        <line x1={30} y1={200} x2={970} y2={200} />
        <line x1={30} y1={315} x2={970} y2={315} />
        <line x1={30} y1={430} x2={970} y2={430} />
        <line x1={30} y1={545} x2={970} y2={545} />

        {/* Longitudes */}
        <line x1={110} y1={40} x2={110} y2={610} />
        <line x1={260} y1={40} x2={260} y2={610} />
        <line x1={420} y1={40} x2={420} y2={610} />
        <line x1={580} y1={40} x2={580} y2={610} />
        <line x1={740} y1={40} x2={740} y2={610} />
        <line x1={900} y1={40} x2={900} y2={610} />
      </g>

      {/* Graticule Degree Annotations */}
      <g fill="#7A5B42" fontSize={8} fontFamily="serif" fontStyle="italic" opacity={0.75}>
        <text x={38} y={82}>50° N</text>
        <text x={38} y={197}>45° N</text>
        <text x={38} y={312}>40° N</text>
        <text x={38} y={427}>35° N</text>
        <text x={38} y={542}>30° N</text>

        <text x={102} y={600} textAnchor="middle">125° W</text>
        <text x={252} y={600} textAnchor="middle">115° W</text>
        <text x={412} y={600} textAnchor="middle">105° W</text>
        <text x={572} y={600} textAnchor="middle">95° W</text>
        <text x={732} y={600} textAnchor="middle">85° W</text>
        <text x={892} y={600} textAnchor="middle">75° W</text>
      </g>

      {/* 3. North America Landmass & USA Mainland Polygonal Coastline */}
      <g className="landmass-layer" filter="url(#carto-shadow)">
        {/* Coastal Outer Shadow Contour */}
        <path
          d={`
            M 60 70
            L 115 80
            Q 135 120 150 145
            Q 132 195 130 230
            Q 120 280 110 340
            Q 100 380 108 420
            Q 130 490 150 540
            Q 158 575 190 580
            Q 225 570 270 560
            L 380 555
            Q 460 550 530 560
            Q 555 580 590 575
            Q 630 560 660 575
            Q 690 585 710 570
            Q 750 560 770 565
            Q 800 575 820 550
            Q 840 540 855 550
            Q 880 575 895 590
            Q 905 570 895 540
            Q 880 500 885 450
            Q 895 420 910 380
            Q 900 340 890 310
            Q 895 280 915 250
            Q 935 220 945 180
            Q 935 150 915 130
            Q 880 110 840 100
            Q 780 80 710 70
            Q 600 65 500 65
            Q 380 65 260 65
            Z
          `}
          fill="none"
          stroke="#C4A87C"
          strokeWidth={6}
          strokeLinejoin="round"
          opacity={0.6}
        />

        {/* Primary Continental Landmass Body */}
        <path
          d={`
            M 60 70
            L 115 80
            Q 135 120 150 145
            Q 132 195 130 230
            Q 120 280 110 340
            Q 100 380 108 420
            Q 130 490 150 540
            Q 158 575 190 580
            Q 225 570 270 560
            L 380 555
            Q 460 550 530 560
            Q 555 580 590 575
            Q 630 560 660 575
            Q 690 585 710 570
            Q 750 560 770 565
            Q 800 575 820 550
            Q 840 540 855 550
            Q 880 575 895 590
            Q 905 570 895 540
            Q 880 500 885 450
            Q 895 420 910 380
            Q 900 340 890 310
            Q 895 280 915 250
            Q 935 220 945 180
            Q 935 150 915 130
            Q 880 110 840 100
            Q 780 80 710 70
            Q 600 65 500 65
            Q 380 65 260 65
            Z
          `}
          fill="url(#land-gradient)"
          stroke="#4D311E"
          strokeWidth={1.8}
          strokeLinejoin="round"
        />

        {/* Coastal Ripple Echo Lines */}
        <path
          d={`
            M 115 80
            Q 135 120 150 145
            Q 132 195 130 230
            Q 120 280 110 340
            Q 100 380 108 420
            Q 130 490 150 540
            Q 158 575 190 580
          `}
          fill="none"
          stroke="rgba(80, 115, 125, 0.45)"
          strokeWidth={1}
          transform="translate(-4, 0)"
        />
        <path
          d={`
            M 660 575
            Q 690 585 710 570
            Q 750 560 770 565
            Q 800 575 820 550
            Q 840 540 855 550
            Q 880 575 895 590
            Q 905 570 895 540
            Q 880 500 885 450
            Q 895 420 910 380
            Q 900 340 890 310
            Q 895 280 915 250
            Q 935 220 945 180
          `}
          fill="none"
          stroke="rgba(80, 115, 125, 0.45)"
          strokeWidth={1}
          transform="translate(4, 0)"
        />
      </g>

      {/* 4. The 5 Great Lakes & St. Lawrence Seaway */}
      <g className="great-lakes-layer">
        {/* Lake Superior */}
        <path
          d="M 625 155 Q 655 135 700 140 Q 735 145 745 165 Q 710 175 665 170 Q 640 175 625 155 Z"
          fill="url(#ocean-gradient)"
          stroke="#4D311E"
          strokeWidth={1.2}
        />
        {/* Lake Michigan */}
        <path
          d="M 708 175 Q 725 180 730 205 Q 735 240 722 255 Q 710 250 705 220 Q 702 195 708 175 Z"
          fill="url(#ocean-gradient)"
          stroke="#4D311E"
          strokeWidth={1.2}
        />
        {/* Lake Huron */}
        <path
          d="M 740 170 Q 765 160 785 175 Q 795 200 780 215 Q 760 210 748 190 Z"
          fill="url(#ocean-gradient)"
          stroke="#4D311E"
          strokeWidth={1.2}
        />
        {/* Lake Erie */}
        <path
          d="M 775 220 Q 805 210 835 225 Q 830 238 800 235 Q 780 235 775 220 Z"
          fill="url(#ocean-gradient)"
          stroke="#4D311E"
          strokeWidth={1.2}
        />
        {/* Lake Ontario */}
        <path
          d="M 825 190 Q 855 185 870 195 Q 865 208 840 205 Q 825 202 825 190 Z"
          fill="url(#ocean-gradient)"
          stroke="#4D311E"
          strokeWidth={1.2}
        />
        {/* St. Lawrence River */}
        <path
          d="M 870 195 Q 890 170 925 140"
          fill="none"
          stroke="#4D311E"
          strokeWidth={1.5}
        />
      </g>

      {/* 5. 19th Century Mountain Range Hachures (Rockies & Appalachians) */}
      <g className="mountain-hachures" stroke="#6E4A28" strokeWidth={1} fill="none" opacity={0.65}>
        {/* Rocky Mountains Chain */}
        <g transform="translate(0, 0)">
          {/* North Section (Calgary/Helena) */}
          <path d="M 270 95 L 280 80 L 290 95 M 285 105 L 295 90 L 305 105 M 345 170 L 355 150 L 365 170 M 360 185 L 370 165 L 380 185" />
          <path d="M 330 215 L 340 195 L 350 215 M 355 240 L 368 220 L 380 240 M 340 265 L 352 245 L 365 265" />
          
          {/* Central Section (Salt Lake / Denver) */}
          <path d="M 320 325 L 332 305 L 345 325 M 340 340 L 355 318 L 370 340 M 475 325 L 490 300 L 505 325 M 495 345 L 510 320 L 525 345" />
          <path d="M 480 375 L 495 350 L 510 375 M 495 405 L 508 385 L 520 405" />

          {/* South Section (Santa Fe) */}
          <path d="M 470 445 L 485 425 L 500 445 M 485 470 L 498 450 L 512 470 M 520 495 L 532 475 L 545 495" />

          {/* Sierra Nevada / Cascades (West Coast) */}
          <path d="M 125 150 L 135 135 L 145 150 M 120 190 L 130 175 L 140 190 M 105 350 L 115 330 L 125 350 M 110 385 L 122 365 L 135 385" />
        </g>

        {/* Appalachian Mountains Chain */}
        <g transform="translate(0, 0)">
          <path d="M 780 320 L 790 305 L 800 320 M 795 300 L 808 280 L 820 300 M 825 285 L 838 265 L 850 285" />
          <path d="M 765 375 L 778 355 L 790 375 M 785 360 L 798 340 L 810 360 M 805 345 L 818 325 L 830 345" />
          <path d="M 755 420 L 768 400 L 780 420 M 775 405 L 788 385 L 800 405" />
        </g>
      </g>

      {/* Mountain Labels */}
      <g fill="#7A5B42" fontSize={9} fontFamily="serif" fontStyle="italic" letterSpacing="0.15em" opacity={0.65}>
        <text x={380} y={230} transform="rotate(45, 380, 230)">ROCKY MOUNTAINS</text>
        <text x={790} y={350} transform="rotate(-40, 790, 350)">APPALACHIAN RIDGE</text>
        <text x={110} y={240} transform="rotate(75, 110, 240)">SIERRA NEVADA</text>
      </g>

      {/* 6. Major State & Regional Boundary Engravings (Dashed Sepia Lines) */}
      <g stroke="rgba(110, 75, 45, 0.22)" strokeWidth="0.8" strokeDasharray="3 2" fill="none">
        {/* US - Canada 49th Parallel Border */}
        <line x1={115} y1={80} x2={625} y2={80} stroke="rgba(110, 75, 45, 0.45)" strokeWidth="1.2" />
        {/* Texas Borders */}
        <path d="M 543 563 L 543 460 L 620 460 L 620 535" />
        {/* California Border */}
        <path d="M 175 190 L 175 400 L 250 510" />
        {/* Mississippi River System */}
        <path
          d="M 645 170 Q 640 250 670 350 Q 690 440 760 540"
          stroke="rgba(80, 115, 125, 0.55)"
          strokeWidth="1.2"
          strokeDasharray="none"
        />
      </g>

      {/* 7. Ocean & Regional Labels */}
      <g fill="#5C7A82" fontSize={11} fontFamily="'Cinzel Decorative', Georgia, serif" fontWeight="700" letterSpacing="0.2em" opacity={0.75}>
        <text x={65} y={480} transform="rotate(-70, 65, 480)">PACIFIC OCEAN</text>
        <text x={930} y={460} transform="rotate(75, 930, 460)">ATLANTIC OCEAN</text>
        <text x={710} y={615} textAnchor="middle">GULF OF MEXICO</text>
      </g>

      {/* 8. Victorian Nautical Compass Rose (in Open Atlantic Corner) */}
      <g transform="translate(890, 480)" className="compass-rose">
        <circle r={36} fill="none" stroke="rgba(140, 99, 5, 0.35)" strokeWidth={1} />
        <circle r={32} fill="none" stroke="rgba(140, 99, 5, 0.6)" strokeWidth={1.5} strokeDasharray="2 2" />
        <circle r={24} fill="rgba(246, 238, 223, 0.6)" stroke="rgba(140, 99, 5, 0.4)" strokeWidth={1} />

        {/* Compass Star Points */}
        {/* North */}
        <polygon points="0,0 -5,-12 0,-30 5,-12" fill="#B8860B" stroke="#6E4E04" strokeWidth={0.5} />
        <polygon points="0,0 0,-30 5,-12" fill="#F6DC88" />
        {/* South */}
        <polygon points="0,0 -5,12 0,30 5,12" fill="#916707" stroke="#6E4E04" strokeWidth={0.5} />
        <polygon points="0,0 0,30 5,12" fill="#E8C96C" />
        {/* East */}
        <polygon points="0,0 12,-5 30,0 12,5" fill="#B8860B" stroke="#6E4E04" strokeWidth={0.5} />
        <polygon points="0,0 30,0 12,5" fill="#F6DC88" />
        {/* West */}
        <polygon points="0,0 -12,-5 -30,0 -12,5" fill="#916707" stroke="#6E4E04" strokeWidth={0.5} />
        <polygon points="0,0 -30,0 -12,5" fill="#E8C96C" />

        {/* Diagonal Points */}
        <polygon points="0,0 -4,-8 -18,-18 -8,-4" fill="#8D4A24" opacity={0.8} />
        <polygon points="0,0 4,-8 18,-18 8,-4" fill="#CD7F32" opacity={0.8} />
        <polygon points="0,0 4,8 18,18 8,4" fill="#8D4A24" opacity={0.8} />
        <polygon points="0,0 -4,8 -18,18 -8,4" fill="#CD7F32" opacity={0.8} />

        {/* Center Rivet */}
        <circle r={4} fill="#382C26" stroke="#F5D77F" strokeWidth={1} />

        {/* Cardinal Letters */}
        <text x={0} y={-34} textAnchor="middle" fontSize={10} fontWeight="bold" fill="#3D2617" fontFamily="serif">N</text>
        <text x={0} y={42} textAnchor="middle" fontSize={9} fontWeight="bold" fill="#7A5B42" fontFamily="serif">S</text>
        <text x={38} y={3} textAnchor="middle" fontSize={9} fontWeight="bold" fill="#7A5B42" fontFamily="serif">E</text>
        <text x={-38} y={3} textAnchor="middle" fontSize={9} fontWeight="bold" fill="#7A5B42" fontFamily="serif">W</text>
      </g>

      {/* 9. Antique Ornamental Cartouche Title Plaque */}
      <g transform="translate(480, 36)" className="survey-cartouche">
        {/* Plaque Background */}
        <rect
          x={-165}
          y={-18}
          width={330}
          height={36}
          rx={6}
          fill="#FAF3E6"
          stroke="#B8860B"
          strokeWidth={1.5}
          filter="url(#carto-shadow)"
        />
        {/* Inner double border */}
        <rect
          x={-161}
          y={-14}
          width={322}
          height={28}
          rx={4}
          fill="none"
          stroke="#4D311E"
          strokeWidth={0.8}
          strokeDasharray="4 2"
        />

        {/* Corner Screws */}
        <circle cx={-156} cy={-9} r={2} fill="#7A5B42" />
        <circle cx={156} cy={-9} r={2} fill="#7A5B42" />
        <circle cx={-156} cy={9} r={2} fill="#7A5B42" />
        <circle cx={156} cy={9} r={2} fill="#7A5B42" />

        {/* Cartouche Title */}
        <text
          x={0}
          y={-1}
          textAnchor="middle"
          fontSize={11}
          fontFamily="'Cinzel Decorative', Georgia, serif"
          fontWeight="700"
          fill="#23140C"
          letterSpacing="0.12em"
        >
          UNITED STATES RAILWAY SURVEY
        </text>
        <text
          x={0}
          y={10}
          textAnchor="middle"
          fontSize={7.5}
          fontFamily="serif"
          fontStyle="italic"
          fill="#785A42"
          letterSpacing="0.18em"
        >
          — CARTOGRAPHICAL REINFORCEMENT LEARNING LAB • 1885 —
        </text>
      </g>

      {/* 10. Outer Brass Bezel Frame & Rivets */}
      <rect
        x={6}
        y={6}
        width={width - 12}
        height={height - 12}
        rx={8}
        fill="none"
        stroke="url(#brass-frame-gradient)"
        strokeWidth={4}
      />
      <rect
        x={11}
        y={11}
        width={width - 22}
        height={height - 22}
        rx={6}
        fill="none"
        stroke="#4D311E"
        strokeWidth={1}
        strokeDasharray="6 3"
      />

      {/* Corner Rivets */}
      <circle cx={16} cy={16} r={3.5} fill="#C59B27" stroke="#382C26" strokeWidth={1} />
      <circle cx={width - 16} cy={16} r={3.5} fill="#C59B27" stroke="#382C26" strokeWidth={1} />
      <circle cx={16} cy={height - 16} r={3.5} fill="#C59B27" stroke="#382C26" strokeWidth={1} />
      <circle cx={width - 16} cy={height - 16} r={3.5} fill="#C59B27" stroke="#382C26" strokeWidth={1} />
    </g>
  );
};

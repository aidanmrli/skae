// Build editable PowerPoint diagram for the Sparse-Koopman multibasin paper.
// All elements are native pptxgenjs shapes/text (no rasterized images),
// so the user can move, edit, and recolor everything in PowerPoint/Keynote.

const pptxgen = require("pptxgenjs");

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 inches
pres.title = "Sparse Koopman Autoencoders for Multibasin Dynamics — overview";
pres.author = "Aidan Li";

const SLIDE_W = 13.333;
const SLIDE_H = 7.5;

// Palette
const C = {
  bg: "F7F6F1",
  card: "FFFFFF",
  cardBorder: "E1E2E1",
  textDark: "1E2761",
  textBody: "1F2937",
  textMuted: "6B7280",
  ruleSoft: "D9D9D6",
  cellInactive: "EDEDE7",
  cellInactiveBorder: "C8C8C0",
  basinA: "6B5B95",   // purple
  basinB: "178582",   // teal
  basinC: "D9755C",   // coral
  ideaBg: "FFF6D6",   // pale yellow
  ideaBorder: "E2C97A",
  ok: "2E7D32",
  bad: "B23A48",
};

// --- Math text helpers ---------------------------------------------------
// Unicode subscript / superscript maps. When a character has a Unicode
// equivalent, we use it inline (string-clean). When it doesn't (e.g. uppercase
// A in subscript), we emit a pptxgenjs rich-text run with subscript:true so
// PowerPoint renders it as a true baseline-shifted glyph.
const SUB = {
  "0":"₀","1":"₁","2":"₂","3":"₃","4":"₄","5":"₅","6":"₆","7":"₇","8":"₈","9":"₉",
  "+":"₊","-":"₋","=":"₌","(":"₍",")":"₎",
  "a":"ₐ","e":"ₑ","h":"ₕ","i":"ᵢ","j":"ⱼ","k":"ₖ","l":"ₗ","m":"ₘ","n":"ₙ",
  "o":"ₒ","p":"ₚ","r":"ᵣ","s":"ₛ","t":"ₜ","u":"ᵤ","v":"ᵥ","x":"ₓ",
};
const SUP = {
  "0":"⁰","1":"¹","2":"²","3":"³","4":"⁴","5":"⁵","6":"⁶","7":"⁷","8":"⁸","9":"⁹",
  "+":"⁺","-":"⁻","=":"⁼","(":"⁽",")":"⁾",
  "a":"ᵃ","b":"ᵇ","c":"ᶜ","d":"ᵈ","e":"ᵉ","f":"ᶠ","g":"ᵍ","h":"ʰ","i":"ⁱ",
  "j":"ʲ","k":"ᵏ","l":"ˡ","m":"ᵐ","n":"ⁿ","o":"ᵒ","p":"ᵖ","r":"ʳ","s":"ˢ",
  "t":"ᵗ","u":"ᵘ","v":"ᵛ","w":"ʷ","x":"ˣ","y":"ʸ","z":"ᶻ",
};
function tryUnicode(map, s) {
  let out = "";
  for (const c of s) {
    if (!(c in map)) return null;
    out += map[c];
  }
  return out;
}

// Parse "x_t", "K^m", "x̂_{t+1}", "K_{S_A}" into a rich-text array.
// baseOpts (italic/color/etc.) are merged into every emitted run.
function math(src, baseOpts = {}) {
  const runs = [];
  function pushPlain(text) {
    if (text) runs.push({ text, options: { ...baseOpts } });
  }
  let i = 0;
  let plainBuf = "";
  while (i < src.length) {
    const ch = src[i];
    if (ch === "_" || ch === "^") {
      const isSub = ch === "_";
      i++;
      let body;
      if (src[i] === "{") {
        const end = src.indexOf("}", i);
        body = src.slice(i + 1, end);
        i = end + 1;
      } else {
        body = src[i] || "";
        i++;
      }
      // Try Unicode substitution first; if it works, keep stream plain.
      const uni = tryUnicode(isSub ? SUB : SUP, body);
      if (uni !== null) {
        plainBuf += uni;
      } else {
        // Flush accumulated plain run, emit a baseline-shifted run.
        pushPlain(plainBuf); plainBuf = "";
        const o = { ...baseOpts };
        if (isSub) o.subscript = true;
        else o.superscript = true;
        runs.push({ text: body, options: o });
      }
    } else {
      plainBuf += ch;
      i++;
    }
  }
  pushPlain(plainBuf);
  return runs;
}

// Convenience: when we just need a flat string (no rich runs needed),
// math() still returns an array — pptxgenjs accepts that everywhere.

const slide = pres.addSlide();
slide.background = { color: C.bg };

// --------- Title bar ---------
slide.addText("Sparse Koopman Autoencoders for Multibasin Dynamical Systems", {
  x: 0.45, y: 0.18, w: 12.4, h: 0.5, margin: 0,
  fontSize: 22, bold: true, fontFace: "DejaVu Serif", color: C.textDark,
  align: "left", valign: "middle",
});
slide.addText(
  "Induced latent sparsity makes the active support a label-free indicator of the relevant basin and local linear law",
  {
    x: 0.45, y: 0.72, w: 12.4, h: 0.30, margin: 0,
    fontSize: 11.5, italic: true, fontFace: "DejaVu Serif", color: C.textMuted,
    align: "left", valign: "middle",
  }
);

// thin divider under title
slide.addShape(pres.shapes.RECTANGLE, {
  x: 0.45, y: 1.06, w: 12.4, h: 0.012,
  fill: { color: C.ruleSoft }, line: { type: "none" },
});

// --------- Three panels ---------
const PANEL_Y = 1.22;
const PANEL_H = 5.7;
const PANEL_W = 4.07;
const PANEL_GAP = 0.22;
const PANEL_X = [
  0.42,
  0.42 + PANEL_W + PANEL_GAP,
  0.42 + 2 * (PANEL_W + PANEL_GAP),
];

function drawCard(x, accentColor) {
  // shadow
  slide.addShape(pres.shapes.RECTANGLE, {
    x: x, y: PANEL_Y, w: PANEL_W, h: PANEL_H,
    fill: { color: C.card },
    line: { color: C.cardBorder, width: 0.75 },
    shadow: { type: "outer", color: "000000", blur: 8, offset: 1, angle: 90, opacity: 0.06 },
  });
  // top accent bar
  slide.addShape(pres.shapes.RECTANGLE, {
    x: x, y: PANEL_Y, w: PANEL_W, h: 0.09,
    fill: { color: accentColor }, line: { type: "none" },
  });
}

function panelHeader(x, accentColor, tag, title) {
  slide.addText(tag, {
    x: x + 0.22, y: PANEL_Y + 0.16, w: PANEL_W - 0.44, h: 0.24, margin: 0,
    fontSize: 9.5, bold: true, fontFace: "DejaVu Serif", color: accentColor,
    charSpacing: 5, align: "left", valign: "middle",
  });
  slide.addText(title, {
    x: x + 0.22, y: PANEL_Y + 0.44, w: PANEL_W - 0.44, h: 0.55, margin: 0,
    fontSize: 12.5, bold: true, fontFace: "DejaVu Serif", color: C.textDark,
    align: "left", valign: "top",
  });
}

drawCard(PANEL_X[0], C.basinC);
panelHeader(PANEL_X[0], C.basinC, "01  ·  THE PROBLEM",
  "Multibasin systems break a single global linearization");

drawCard(PANEL_X[1], C.textDark);
panelHeader(PANEL_X[1], C.textDark, "02  ·  THE MODEL",
  "Sparse-coding Koopman autoencoder");

drawCard(PANEL_X[2], C.basinB);
panelHeader(PANEL_X[2], C.basinB, "03  ·  THE KEY IDEA",
  "The support identifies the basin and the local law");

// =========================================================
// PANEL A — Problem
// =========================================================
{
  const px = PANEL_X[0];
  const cx = px + PANEL_W / 2;

  // State-space sketch background
  const sceneX = px + 0.32, sceneY = 2.30, sceneW = PANEL_W - 0.64, sceneH = 2.20;
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: sceneX, y: sceneY, w: sceneW, h: sceneH,
    fill: { color: "FBFAF4" },
    line: { color: C.cardBorder, width: 0.5, dashType: "dash" },
    rectRadius: 0.08,
  });
  slide.addText("state space", {
    x: sceneX + 0.08, y: sceneY + 0.04, w: 1.4, h: 0.22, margin: 0,
    fontSize: 9, italic: true, fontFace: "DejaVu Serif", color: C.textMuted,
  });

  // helper: draw a "well" with attractor + a few inward arrows
  function drawBasin(bx, by, bw, color, label) {
    // outer halo
    slide.addShape(pres.shapes.OVAL, {
      x: bx, y: by, w: bw, h: bw,
      fill: { color: color, transparency: 80 },
      line: { color: color, width: 1, dashType: "dash" },
    });
    // mid ring
    slide.addShape(pres.shapes.OVAL, {
      x: bx + bw * 0.18, y: by + bw * 0.18, w: bw * 0.64, h: bw * 0.64,
      fill: { color: color, transparency: 65 },
      line: { type: "none" },
    });
    // attractor dot
    const dotW = bw * 0.18;
    slide.addShape(pres.shapes.OVAL, {
      x: bx + bw / 2 - dotW / 2, y: by + bw / 2 - dotW / 2, w: dotW, h: dotW,
      fill: { color: color }, line: { type: "none" },
    });
    // inward arrows (simple lines from outside)
    const cx2 = bx + bw / 2, cy2 = by + bw / 2;
    const ar = bw * 0.42, ir = bw * 0.22;
    const angles = [Math.PI * 0.25, Math.PI * 0.75, Math.PI * 1.25, Math.PI * 1.75];
    angles.forEach(a => {
      const x1 = cx2 + ar * Math.cos(a), y1 = cy2 + ar * Math.sin(a);
      const x2 = cx2 + ir * Math.cos(a), y2 = cy2 + ir * Math.sin(a);
      slide.addShape(pres.shapes.LINE, {
        x: Math.min(x1, x2), y: Math.min(y1, y2),
        w: Math.abs(x2 - x1), h: Math.abs(y2 - y1),
        line: { color: color, width: 1, endArrowType: "triangle",
                beginArrowType: "none" },
        flipH: x2 < x1, flipV: y2 < y1,
      });
    });
    // label
    slide.addText(label, {
      x: bx, y: by + bw + 0.02, w: bw, h: 0.22, margin: 0,
      fontSize: 9, bold: true, fontFace: "DejaVu Serif", color: color,
      align: "center",
    });
  }

  const bw = 0.62;
  drawBasin(sceneX + 0.18, sceneY + 0.18, bw, C.basinA, "Basin A");
  drawBasin(sceneX + sceneW - 0.18 - bw, sceneY + 0.20, bw, C.basinB, "Basin B");
  drawBasin(sceneX + sceneW / 2 - bw / 2, sceneY + sceneH - bw - 0.30, bw, C.basinC, "Basin C");

  // Two takeaway rows below scene (shape-based icons — font-independent)
  const takeY = 4.85;
  function drawCrossIcon(cx0, cy0, r, color) {
    const d = r * 0.65;
    slide.addShape(pres.shapes.LINE, {
      x: cx0 - d, y: cy0 - d, w: 2 * d, h: 2 * d,
      line: { color: "FFFFFF", width: 2.25, endArrowType: "none" },
    });
    slide.addShape(pres.shapes.LINE, {
      x: cx0 - d, y: cy0 - d, w: 2 * d, h: 2 * d,
      line: { color: "FFFFFF", width: 2.25, endArrowType: "none" },
      flipV: true,
    });
  }
  function drawCheckIcon(cx0, cy0, r, color) {
    const a = r * 0.55;
    // short stroke (down-right)
    slide.addShape(pres.shapes.LINE, {
      x: cx0 - a, y: cy0 - 0.01, w: a * 0.55, h: a * 0.7,
      line: { color: "FFFFFF", width: 2.25, endArrowType: "none" },
    });
    // long stroke (up-right)
    slide.addShape(pres.shapes.LINE, {
      x: cx0 - a * 0.45, y: cy0 - a * 0.85, w: a * 1.35, h: a * 1.55,
      line: { color: "FFFFFF", width: 2.25, endArrowType: "none" },
      flipV: true,
    });
  }
  function takeaway(y, kind, markColor, text) {
    const r = 0.13;
    const cx0 = px + 0.32 + r, cy0 = y + r;
    slide.addShape(pres.shapes.OVAL, {
      x: px + 0.32, y: y, w: 2 * r, h: 2 * r,
      fill: { color: markColor }, line: { type: "none" },
    });
    if (kind === "x") drawCrossIcon(cx0, cy0, r, markColor);
    else drawCheckIcon(cx0, cy0, r, markColor);
    slide.addText(text, {
      x: px + 0.66, y: y - 0.04, w: PANEL_W - 0.66 - 0.32, h: 0.6, margin: 0,
      fontSize: 11.5, fontFace: "DejaVu Serif", color: C.textBody,
      align: "left", valign: "top",
    });
  }
  takeaway(takeY, "x", C.bad,
    "A single global linear K cannot describe all basins simultaneously.");
  takeaway(takeY + 0.78, "check", C.ok,
    "Each basin admits a local linear description, but they are mutually incompatible.");
}

// =========================================================
// PANEL B — Model (vertical pipeline)
// =========================================================
{
  const px = PANEL_X[1];
  const cx = px + PANEL_W / 2;

  // helpers
  function smallBox(x, y, w, h, label, opts = {}) {
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x, y, w, h,
      fill: { color: opts.fill || "F4F4EE" },
      line: { color: opts.border || C.cardBorder, width: 0.75 },
      rectRadius: 0.05,
    });
    slide.addText(label, {
      x, y, w, h, margin: 0,
      fontSize: opts.fontSize || 11, bold: opts.bold !== false,
      fontFace: "DejaVu Serif", color: opts.color || C.textDark,
      align: "center", valign: "middle",
    });
  }

  function downArrow(x, y, h, label) {
    slide.addShape(pres.shapes.DOWN_ARROW, {
      x: x - 0.07, y: y, w: 0.14, h: h,
      fill: { color: C.textMuted }, line: { type: "none" },
    });
    if (label) {
      slide.addText(label, {
        x: x + 0.10, y: y + h / 2 - 0.13, w: 0.6, h: 0.26, margin: 0,
        fontSize: 11, bold: true, fontFace: "DejaVu Serif", italic: true,
        color: C.textDark, align: "left", valign: "middle",
      });
    }
  }

  // --- pipeline geometry (tight, leaves room for bottom callout) ---
  let y = 2.30;

  // x_t
  smallBox(cx - 0.34, y, 0.68, 0.32, math("x_t", { italic: true }),
           { fontSize: 12 });
  y += 0.32 + 0.04;
  downArrow(cx, y, 0.18);
  y += 0.18 + 0.04;

  // Encoder — rich text mixing roman label and italic Eθ
  smallBox(cx - 1.05, y, 2.10, 0.40, [
    { text: "Encoder  ", options: { bold: true } },
    { text: "Eθ", options: { italic: true, bold: true } },
    { text: "   (LISTA-style sparse code)", options: { bold: true } },
  ], { fontSize: 10 });
  y += 0.40 + 0.04;
  downArrow(cx, y, 0.18);
  y += 0.18 + 0.04;

  // z_t cells
  const NCELLS = 10;
  const cellW = 0.17, cellH = 0.28, cellGap = 0.02;
  const stripW = NCELLS * cellW + (NCELLS - 1) * cellGap;
  const supportA = new Set([1, 3, 7]); // 0-indexed
  function drawStrip(yTop, supportSet, color, label) {
    const x0 = cx - stripW / 2;
    for (let i = 0; i < NCELLS; i++) {
      const cellX = x0 + i * (cellW + cellGap);
      const isOn = supportSet.has(i);
      slide.addShape(pres.shapes.RECTANGLE, {
        x: cellX, y: yTop, w: cellW, h: cellH,
        fill: { color: isOn ? color : C.cellInactive },
        line: { color: isOn ? color : C.cellInactiveBorder, width: 0.5 },
      });
    }
    if (label) {
      slide.addText(label, {
        x: cx - 1.7, y: yTop + cellH + 0.01, w: 3.4, h: 0.20, margin: 0,
        fontSize: 9, italic: true, fontFace: "DejaVu Serif",
        color: C.textMuted, align: "center",
      });
    }
  }

  drawStrip(y, supportA, C.basinA,
    "zₜ  —  sparse latent code, active support S(xₜ)");
  y += cellH + 0.20;

  // K transition arrow (with K label)
  downArrow(cx, y, 0.26, "× K");
  y += 0.26 + 0.04;

  // ẑ_{t+1} cells
  const supportA2 = new Set([1, 3, 7]);
  drawStrip(y, supportA2, C.basinA,
    "ẑₜ₊₁ = K · zₜ   (linear advance within S)");
  y += cellH + 0.20;

  downArrow(cx, y, 0.18);
  y += 0.18 + 0.04;

  // Decoder
  smallBox(cx - 0.95, y, 1.90, 0.40, [
    { text: "Decoder  ", options: { bold: true } },
    { text: "D", options: { italic: true, bold: true } },
    { text: "  (linear)", options: { bold: true } },
  ], { fontSize: 10 });
  y += 0.40 + 0.04;
  downArrow(cx, y, 0.18);
  y += 0.18 + 0.04;

  // x̂_{t+1}
  smallBox(cx - 0.42, y, 0.84, 0.32, math("x̂_{t+1}", { italic: true }),
           { fontSize: 12 });
  y += 0.32;

  // bottom callout: periodic re-encoding
  const noteY = PANEL_Y + PANEL_H - 0.70;
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: px + 0.30, y: noteY, w: PANEL_W - 0.60, h: 0.55,
    fill: { color: "F0EFE7" }, line: { color: C.cardBorder, width: 0.5 },
    rectRadius: 0.05,
  });
  slide.addText([
    { text: "Periodic re-encoding.  ",
      options: { bold: true, color: C.textDark } },
    { text: "Decode and re-encode every m steps so the support can change as the predicted state moves.",
      options: { color: C.textBody } },
  ], {
    x: px + 0.40, y: noteY, w: PANEL_W - 0.80, h: 0.55, margin: 0,
    fontSize: 10, fontFace: "DejaVu Serif", align: "left", valign: "middle",
  });
}

// =========================================================
// PANEL C — Idea
// =========================================================
{
  const px = PANEL_X[2];

  const NCELLS = 8;
  const cellW = 0.16, cellH = 0.26, cellGap = 0.02;
  const stripW = NCELLS * cellW + (NCELLS - 1) * cellGap;

  function drawSupportRow(yRow, basinColor, basinLabel, supportSet, lawLabel, highlightCols) {
    // basin badge with state dot (left)
    const badgeX = px + 0.22, badgeY = yRow + 0.04;
    const badgeW = 0.85, badgeH = 0.46;
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: badgeX, y: badgeY, w: badgeW, h: badgeH,
      fill: { color: basinColor, transparency: 82 },
      line: { color: basinColor, width: 0.75 },
      rectRadius: 0.06,
    });
    // attractor dot in badge
    slide.addShape(pres.shapes.OVAL, {
      x: badgeX + 0.10, y: badgeY + badgeH / 2 - 0.10, w: 0.20, h: 0.20,
      fill: { color: basinColor }, line: { type: "none" },
    });
    slide.addText(basinLabel, {
      x: badgeX + 0.34, y: badgeY, w: badgeW - 0.36, h: badgeH, margin: 0,
      fontSize: 10, bold: true, fontFace: "DejaVu Serif", color: basinColor,
      align: "left", valign: "middle",
    });

    // arrow 1
    slide.addShape(pres.shapes.RIGHT_ARROW, {
      x: badgeX + badgeW + 0.03, y: badgeY + badgeH / 2 - 0.07, w: 0.16, h: 0.14,
      fill: { color: C.textMuted }, line: { type: "none" },
    });

    // support strip
    const stripX = badgeX + badgeW + 0.24;
    const stripY = badgeY + badgeH / 2 - cellH / 2;
    for (let i = 0; i < NCELLS; i++) {
      const cellX = stripX + i * (cellW + cellGap);
      const isOn = supportSet.has(i);
      slide.addShape(pres.shapes.RECTANGLE, {
        x: cellX, y: stripY, w: cellW, h: cellH,
        fill: { color: isOn ? basinColor : C.cellInactive },
        line: { color: isOn ? basinColor : C.cellInactiveBorder, width: 0.5 },
      });
    }
    // strip label below
    slide.addText("active support  S(x)", {
      x: stripX - 0.20, y: stripY + cellH + 0.02, w: stripW + 0.40, h: 0.20, margin: 0,
      fontSize: 8.5, italic: true, fontFace: "DejaVu Serif",
      color: C.textMuted, align: "center",
    });

    // arrow 2
    const arr2X = stripX + stripW + 0.04;
    slide.addShape(pres.shapes.RIGHT_ARROW, {
      x: arr2X, y: badgeY + badgeH / 2 - 0.07, w: 0.16, h: 0.14,
      fill: { color: C.textMuted }, line: { type: "none" },
    });

    // mini K matrix box with selected columns
    const Kx = arr2X + 0.20, Ky = badgeY - 0.02, Kw = 0.55, Kh = 0.55;
    slide.addShape(pres.shapes.RECTANGLE, {
      x: Kx, y: Ky, w: Kw, h: Kh,
      fill: { color: "F4F4EE" }, line: { color: C.cardBorder, width: 0.6 },
    });
    // 4-column mini matrix: highlight 2 columns in basin color
    const nC = 4;
    const colW = (Kw - 0.04) / nC;
    const matY = Ky + 0.04, matH = Kh - 0.08;
    const cols = highlightCols || [0, 2];
    for (let c = 0; c < nC; c++) {
      const cX = Kx + 0.02 + c * colW;
      const isOn = cols.includes(c);
      slide.addShape(pres.shapes.RECTANGLE, {
        x: cX, y: matY, w: colW * 0.85, h: matH,
        fill: { color: isOn ? basinColor : "E5E5DE",
                transparency: isOn ? 0 : 0 },
        line: { type: "none" },
      });
    }
    // label below mini K
    slide.addText(lawLabel, {
      x: Kx - 0.20, y: Ky + Kh + 0.02, w: Kw + 0.40, h: 0.22, margin: 0,
      fontSize: 9.5, italic: true, bold: true, fontFace: "DejaVu Serif",
      color: basinColor, align: "center",
    });
  }

  // Row 1
  drawSupportRow(2.30, C.basinA, "Basin A",
    new Set([1, 3, 6]), math("K_A", { italic: true, bold: true }), [0, 2]);
  // Row 2
  drawSupportRow(3.20, C.basinB, "Basin B",
    new Set([0, 4, 7]), math("K_B", { italic: true, bold: true }), [1, 3]);

  // Divider line
  slide.addShape(pres.shapes.LINE, {
    x: px + 0.30, y: 4.18, w: PANEL_W - 0.60, h: 0,
    line: { color: C.cardBorder, width: 0.6, dashType: "dash" },
  });

  // Mechanism explanation
  slide.addText([
    { text: "Different basin  ",
      options: { bold: true, color: C.textDark } },
    { text: "→  ", options: { color: C.textMuted } },
    { text: "different support  ",
      options: { bold: true, color: C.textDark } },
    { text: "→  ", options: { color: C.textMuted } },
    { text: "different active columns of K  ",
      options: { bold: true, color: C.textDark } },
    { text: "→  ", options: { color: C.textMuted } },
    { text: "different effective local linear law.",
      options: { color: C.textBody } },
  ], {
    x: px + 0.28, y: 4.28, w: PANEL_W - 0.56, h: 0.78, margin: 0,
    fontSize: 10.5, fontFace: "DejaVu Serif", align: "left", valign: "top",
  });

  // Re-encoding refresh visual
  const reY = 5.20;
  slide.addText("Periodic re-encoding refreshes the support across regimes:", {
    x: px + 0.28, y: reY, w: PANEL_W - 0.56, h: 0.24, margin: 0,
    fontSize: 9.5, italic: true, fontFace: "DejaVu Serif", color: C.textMuted,
  });

  // small sequence:  z_A  →  K^m  →  decode  →  encode  →  z_B
  const seqY = reY + 0.30;
  const seqH = 0.36;
  function seqBox(x, w, label, color) {
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x, y: seqY, w, h: seqH,
      fill: { color: color || "F4F4EE" },
      line: { color: color ? color : C.cardBorder, width: 0.6 },
      rectRadius: 0.04,
    });
    slide.addText(label, {
      x, y: seqY, w, h: seqH, margin: 0,
      fontSize: 9.5, bold: true, fontFace: "DejaVu Serif",
      color: color ? "FFFFFF" : C.textDark,
      align: "center", valign: "middle",
    });
  }
  function seqArrow(x) {
    slide.addShape(pres.shapes.RIGHT_ARROW, {
      x: x, y: seqY + seqH / 2 - 0.07, w: 0.14, h: 0.14,
      fill: { color: C.textMuted }, line: { type: "none" },
    });
  }
  let sx = px + 0.28;
  seqBox(sx, 0.42,
    math("z_A", { italic: true, bold: true, color: "FFFFFF" }),
    C.basinA); sx += 0.42 + 0.03;
  seqArrow(sx); sx += 0.16;
  seqBox(sx, 0.46, "Kᵐ"); sx += 0.46 + 0.03;
  seqArrow(sx); sx += 0.16;
  seqBox(sx, 0.55, "decode"); sx += 0.55 + 0.03;
  seqArrow(sx); sx += 0.16;
  seqBox(sx, 0.55, "encode"); sx += 0.55 + 0.03;
  seqArrow(sx); sx += 0.16;
  seqBox(sx, 0.42,
    math("z_B", { italic: true, bold: true, color: "FFFFFF" }),
    C.basinB);

  // bottom note
  slide.addText(
    "→  the support can switch when the trajectory crosses into another basin.",
    {
      x: px + 0.28, y: seqY + seqH + 0.08, w: PANEL_W - 0.56, h: 0.30, margin: 0,
      fontSize: 9.5, italic: true, fontFace: "DejaVu Serif",
      color: C.textBody, align: "left",
    }
  );
}

// =========================================================
// Footer band (Core idea)
// =========================================================
{
  const fy = 7.05;
  const fh = 0.34;
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 0.42, y: fy, w: SLIDE_W - 0.84, h: fh,
    fill: { color: C.ideaBg },
    line: { color: C.ideaBorder, width: 0.6 },
    rectRadius: 0.06,
  });
  slide.addText([
    { text: "Core idea.  ", options: { bold: true, color: "8A6C00" } },
    { text: "Sparse latent code  ", options: { color: C.textDark, bold: true } },
    { text: "→  ", options: { color: C.textMuted } },
    { text: "active support agrees with basin  ", options: { color: C.textDark, bold: true } },
    { text: "→  ", options: { color: C.textMuted } },
    { text: "support routes prediction to a local linear law", options: { color: C.textDark, bold: true } },
    { text: "    (no basin labels used during training).",
      options: { italic: true, color: C.textMuted } },
  ], {
    x: 0.55, y: fy, w: SLIDE_W - 1.10, h: fh, margin: 0,
    fontSize: 11, fontFace: "DejaVu Serif", align: "left", valign: "middle",
  });
}

const outPath = "/home/mila/l/lia/skae/docs/figures/neurips_paper_2026/paper_overview_diagram.pptx";
pres.writeFile({ fileName: outPath })
  .then((p) => console.log("WROTE:", p))
  .catch((e) => { console.error(e); process.exit(1); });

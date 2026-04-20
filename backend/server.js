/**
 * Market Fit Predictor — Node.js/Express Backend
 * Port: 5000
 */

require("dotenv").config();
const express   = require("express");
const cors      = require("cors");
const morgan    = require("morgan");
const mongoose  = require("mongoose");
const jwt       = require("jsonwebtoken");
const axios     = require("axios");
const rateLimit = require("express-rate-limit");
const { body, validationResult } = require("express-validator");

const { User, Product }         = require("./models");
const { getAIRecommendation }   = require("./aiRecommendation");

const app  = express();
const PORT = process.env.PORT || 5000;

// ─── Middleware ───────────────────────────────────────────────────────────────
app.use(cors({ origin: "*", credentials: true }));
app.use(express.json());
app.use(morgan("dev"));

const limiter = rateLimit({ windowMs: 15 * 60 * 1000, max: 100 });
app.use(limiter);

// ─── MongoDB ──────────────────────────────────────────────────────────────────
mongoose
  .connect(process.env.MONGO_URI || "mongodb://localhost:27017/market_fit_db")
  .then(() => console.log("✅ MongoDB connected"))
  .catch((err) => console.error("❌ MongoDB error:", err.message));

// ─── Auth middleware ──────────────────────────────────────────────────────────
function authenticate(req, res, next) {
  const header = req.headers.authorization;
  if (!header?.startsWith("Bearer "))
    return res.status(401).json({ error: "No token provided" });
  try {
    const decoded = jwt.verify(header.split(" ")[1], process.env.JWT_SECRET || "secret");
    req.userId = decoded.id;
    next();
  } catch {
    res.status(401).json({ error: "Invalid or expired token" });
  }
}

// ─── Validation helper ────────────────────────────────────────────────────────
function validateRequest(req, res) {
  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    res.status(422).json({ errors: errors.array() });
    return false;
  }
  return true;
}

// ─── Category feature map ─────────────────────────────────────────────────────
const CATEGORY_FEATURES = {
  smartphone:       ["ram_gb", "storage_gb", "battery_mah", "camera_mp", "display_size"],
  laptop:           ["ram_gb", "storage_gb", "processor_ghz", "battery_wh"],
  smartwatch:       ["battery_life_days", "display_size_mm", "resolution_px", "connectivity_count"],
  washing_machine:  ["capacity_kg", "spin_speed_rpm", "energy_rating", "water_consumption_l"],
};

// ══════════════════════════════════════════════════════════════════════════════
// AUTH ROUTES
// ══════════════════════════════════════════════════════════════════════════════

// POST /register
app.post(
  "/register",
  [
    body("name").trim().notEmpty().withMessage("Name is required"),
    body("email").isEmail().withMessage("Valid email required"),
    body("password").isLength({ min: 6 }).withMessage("Password min 6 chars"),
  ],
  async (req, res) => {
    if (!validateRequest(req, res)) return;
    const { name, email, password } = req.body;
    try {
      if (await User.findOne({ email }))
        return res.status(409).json({ error: "Email already registered" });
      const user  = await User.create({ name, email, password });
      const token = jwt.sign({ id: user._id }, process.env.JWT_SECRET || "secret", {
        expiresIn: process.env.JWT_EXPIRES_IN || "7d",
      });
      res.status(201).json({ token, user: user.toSafeObject() });
    } catch (err) {
      res.status(500).json({ error: err.message });
    }
  }
);

// POST /login
app.post(
  "/login",
  [
    body("email").isEmail().withMessage("Valid email required"),
    body("password").notEmpty().withMessage("Password required"),
  ],
  async (req, res) => {
    if (!validateRequest(req, res)) return;
    const { email, password } = req.body;
    try {
      const user = await User.findOne({ email });
      if (!user || !(await user.comparePassword(password)))
        return res.status(401).json({ error: "Invalid credentials" });
      const token = jwt.sign({ id: user._id }, process.env.JWT_SECRET || "secret", {
        expiresIn: process.env.JWT_EXPIRES_IN || "7d",
      });
      res.json({ token, user: user.toSafeObject() });
    } catch (err) {
      res.status(500).json({ error: err.message });
    }
  }
);

// ══════════════════════════════════════════════════════════════════════════════
// PREDICTION ROUTE
// ══════════════════════════════════════════════════════════════════════════════

// POST /predict-product
app.post("/predict-product", authenticate, async (req, res) => {
  const { category, specifications } = req.body;

  if (!category || !specifications)
    return res.status(400).json({ error: "category and specifications are required" });

  const cat = category.toLowerCase().trim().replace(" ", "_");
  if (!CATEGORY_FEATURES[cat])
    return res.status(400).json({ error: `Unknown category: ${category}` });

  try {
    // ── 1. Call FastAPI ML service ──────────────────────────────────────────
    const fastapiUrl = `${process.env.FASTAPI_URL || "http://localhost:8000"}/predict`;
    let mlResponse;
    try {
      const { data } = await axios.post(fastapiUrl, {
        category: cat,
        features: specifications,
      }, { timeout: 10000 });
      mlResponse = data;
    } catch (mlErr) {
      console.error("[ML] FastAPI error:", mlErr.message);
      return res.status(502).json({ error: "ML service unavailable. Is FastAPI running?" });
    }

    const score = mlResponse.market_fit_score;

    // ── 2. Get AI recommendation ────────────────────────────────────────────
    let aiRec = null;
    try {
      aiRec = await getAIRecommendation(cat, specifications, score);
    } catch (aiErr) {
      console.error("[AI] Recommendation error:", aiErr.message);
      aiRec = {
  recommendation: "AI analysis unavailable. Please check your API configuration."
};
    }

    // ── 3. Save to MongoDB ──────────────────────────────────────────────────
    const product = await Product.create({
      userId:           req.userId,
      category:         cat,
      specifications:   new Map(Object.entries(specifications)),
      market_fit_score: score,
      ai_recommendation: aiRec,
    });

    res.json({
      product_id:        product._id,
      category:          cat,
      specifications,
      market_fit_score:  score,
      ai_recommendation: aiRec,
    });
  } catch (err) {
    console.error("[Predict]", err);
    res.status(500).json({ error: err.message });
  }
});

// ══════════════════════════════════════════════════════════════════════════════
// PRODUCT HISTORY
// ══════════════════════════════════════════════════════════════════════════════

// GET /user-products
app.get("/user-products", authenticate, async (req, res) => {
  try {
    const products = await Product.find({ userId: req.userId })
      .sort({ createdAt: -1 })
      .limit(20)
      .lean();

    // Convert Map to plain object for JSON serialization
    const serialized = products.map((p) => ({
      ...p,
      specifications: p.specifications instanceof Map
        ? Object.fromEntries(p.specifications)
        : p.specifications,
    }));

    res.json({ products: serialized, count: serialized.length });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// GET /health
app.get("/health", (req, res) => {
  res.json({ status: "ok", db: mongoose.connection.readyState === 1 ? "connected" : "disconnected" });
});

// ─── Start ────────────────────────────────────────────────────────────────────
app.listen(PORT, () => {
  console.log(`🚀 Backend running at http://localhost:${PORT}`);
});

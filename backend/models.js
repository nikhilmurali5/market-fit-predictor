const mongoose = require("mongoose");
const bcrypt   = require("bcryptjs");

// ─── User ─────────────────────────────────────────────────────────────────────
const userSchema = new mongoose.Schema(
  {
    name:     { type: String, required: true, trim: true },
    email:    { type: String, required: true, unique: true, lowercase: true, trim: true },
    password: { type: String, required: true, minlength: 6 },
  },
  { timestamps: true }
);

userSchema.pre("save", async function (next) {
  if (!this.isModified("password")) return next();
  this.password = await bcrypt.hash(this.password, 12);
  next();
});

userSchema.methods.comparePassword = function (candidate) {
  return bcrypt.compare(candidate, this.password);
};

userSchema.methods.toSafeObject = function () {
  const obj = this.toObject();
  delete obj.password;
  return obj;
};

// ─── Product ──────────────────────────────────────────────────────────────────
const productSchema = new mongoose.Schema(
  {
    userId:   { type: mongoose.Schema.Types.ObjectId, ref: "User", required: true, index: true },
    category: { type: String, required: true, enum: ["smartphone", "laptop", "smartwatch", "washing_machine"] },
    specifications: { type: Map, of: Number, required: true },
    market_fit_score: { type: Number, min: 0, max: 100 },
    ai_recommendation: {
      best_use_case:  String,
      advantages:     [String],
      disadvantages:  [String],
      suggestions:    [String],
      raw_response:   String,
    },
  },
  { timestamps: true }
);

const User    = mongoose.model("User",    userSchema);
const Product = mongoose.model("Product", productSchema);

module.exports = { User, Product };

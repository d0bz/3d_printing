#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SDK_ROOT="/tmp/android-sdk"
BUILD_TOOLS="$SDK_ROOT/build-tools/34.0.0"
PLATFORM="$SDK_ROOT/platforms/android-34/android.jar"

OUT_DIR="$PROJECT_DIR/build"
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR/classes" "$OUT_DIR/dex"

echo "[1/5] Compiling Java sources..."
javac --release 11 \
    -classpath "$PLATFORM" \
    -d "$OUT_DIR/classes" \
    "$PROJECT_DIR/src/com/printguard/cam/CameraServerService.java" \
    "$PROJECT_DIR/src/com/printguard/cam/MainActivity.java"

echo "[2/5] Converting bytecode to DEX (d8)..."
jar cf "$OUT_DIR/classes.jar" -C "$OUT_DIR/classes" .
"$BUILD_TOOLS/d8" --lib "$PLATFORM" --min-api 24 --output "$OUT_DIR/dex" "$OUT_DIR/classes.jar"

echo "[3/5] Packaging AndroidManifest.xml with aapt2..."
"$BUILD_TOOLS/aapt2" link \
    -o "$OUT_DIR/unsigned.apk" \
    -I "$PLATFORM" \
    --manifest "$PROJECT_DIR/AndroidManifest.xml"

echo "[4/5] Adding classes.dex and aligning APK..."
(cd "$OUT_DIR/dex" && zip -q "$OUT_DIR/unsigned.apk" classes.dex)
"$BUILD_TOOLS/zipalign" -f -p 4 "$OUT_DIR/unsigned.apk" "$OUT_DIR/aligned.apk"

KEYSTORE="$PROJECT_DIR/printguard-release.jks"
if [ ! -f "$KEYSTORE" ]; then
    keytool -genkeypair \
        -keystore "$KEYSTORE" \
        -alias printguard \
        -keyalg RSA \
        -keysize 2048 \
        -validity 10000 \
        -storepass printguard123 \
        -keypass printguard123 \
        -dname "CN=PrintGuard, OU=3DPrint, O=PrintGuard, L=Tallinn, S=Harju, C=EE"
fi

echo "[5/5] Signing APK (v1 + v2 + v3 signatures)..."
FINAL_APK="$PROJECT_DIR/PrintGuardCam.apk"
"$BUILD_TOOLS/apksigner" sign \
    --v1-signing-enabled true \
    --v2-signing-enabled true \
    --v3-signing-enabled true \
    --ks "$KEYSTORE" \
    --ks-key-alias printguard \
    --ks-pass pass:printguard123 \
    --key-pass pass:printguard123 \
    --out "$FINAL_APK" \
    "$OUT_DIR/aligned.apk"

"$BUILD_TOOLS/apksigner" verify --verbose "$FINAL_APK" | head -n 10
ls -lh "$FINAL_APK"

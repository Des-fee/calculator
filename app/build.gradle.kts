plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("com.chaquo.python")
}

// CI passes the build number; local builds use 1.
val buildNumber = (System.getenv("BUILD_NUMBER") ?: "1").toInt()

android {
    namespace = "io.github.desfee.calculator"
    compileSdk = 34

    defaultConfig {
        applicationId = "io.github.desfee.calculator"
        minSdk = 24
        targetSdk = 34
        versionCode = buildNumber
        versionName = "1.0.$buildNumber"
        ndk {
            // phones are arm64; CI emulator tests pass -Pabis=x86_64
            abiFilters += ((project.findProperty("abis") as String?) ?: "arm64-v8a").split(",")
        }
    }

    // The release key is decrypted by CI from keystore/release.jks.enc (secret SIGNING_PASSWORD).
    // Without it (local builds) the APK is signed with the debug key.
    val releaseKey = System.getenv("KEYSTORE_FILE")?.let { file(it) }?.takeIf { it.exists() }
    signingConfigs {
        if (releaseKey != null) {
            create("release") {
                storeFile = releaseKey
                storePassword = System.getenv("KEYSTORE_PASSWORD")
                keyAlias = System.getenv("KEY_ALIAS") ?: "calculator"
                keyPassword = System.getenv("KEY_PASSWORD") ?: System.getenv("KEYSTORE_PASSWORD")
            }
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            signingConfig = signingConfigs.findByName("release") ?: signingConfigs.getByName("debug")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    androidResources {
        noCompress += "onnx"
    }
}

chaquopy {
    defaultConfig {
        version = "3.11"
        pip {
            install("sympy==1.13.3")
        }
    }
}

dependencies {
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.20.0")
    // typeset formulas (LaTeX) for answers and recognised input
    implementation("ru.noties:jlatexmath-android:0.2.0")
    implementation("ru.noties:jlatexmath-android-font-cyrillic:0.2.0")
}

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
            abiFilters += listOf("arm64-v8a", "armeabi-v7a")
        }
    }

    signingConfigs {
        create("release") {
            // The keystore lives in the private repository; CI can override it with secrets.
            storeFile = file(System.getenv("KEYSTORE_FILE") ?: "../keystore/release.jks")
            storePassword = System.getenv("KEYSTORE_PASSWORD") ?: "calc-release-2026"
            keyAlias = System.getenv("KEY_ALIAS") ?: "calculator"
            keyPassword = System.getenv("KEY_PASSWORD") ?: "calc-release-2026"
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            signingConfig = signingConfigs.getByName("release")
        }
        debug {
            signingConfig = signingConfigs.getByName("release")
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
}

package com.example.ecm

import android.app.Application

class EcmApplication : Application() {
    override fun onCreate() {
        super.onCreate()
        AppSettings.applyTheme(this)
    }
}

package com.example.ecm

/** Receives ECM log lines from native code while a run is in progress. */
fun interface EcmLogCallback {
    fun onLine(line: String)
}

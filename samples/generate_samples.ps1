# Generates 5 spoken WAV test files using the Windows built-in TTS engine.
# Run from anywhere:  powershell -File samples\generate_samples.ps1
# Output: 16 kHz PCM WAV files in this folder, ready for faster-whisper.

Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$outDir = Split-Path -Parent $MyInvocation.MyCommand.Path

$samples = @{
    "sample1_greeting.wav"  = "Hello, my name is Alex. I am testing the speech to text and large language model pipeline."
    "sample2_question.wav"  = "What is the capital of France, and can you tell me three interesting facts about it?"
    "sample3_meeting.wav"   = "In today's meeting we discussed the project timeline, the budget for the next quarter, and the new hiring plan."
    "sample4_weather.wav"   = "The weather forecast for tomorrow predicts heavy rain in the morning followed by clear skies in the afternoon."
    "sample5_tech.wav"      = "Artificial intelligence is transforming healthcare, finance, and transportation through machine learning and automation."
}

foreach ($name in $samples.Keys) {
    $path = Join-Path $outDir $name
    $synth.SetOutputToWaveFile($path)
    $synth.Speak($samples[$name])
    Write-Host "Wrote $name"
}

$synth.SetOutputToDefaultAudioDevice()
$synth.Dispose()
Write-Host "Done. 5 WAV files created in $outDir"

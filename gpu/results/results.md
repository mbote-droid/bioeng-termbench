GPU: Tesla T4  CPU: x86_64

| backend | dtype | n | cells | M cell-updates/s | speed-up vs NumPy |
|---|---|---|---|---|---|
| numpy | float64 | 64 | 274,625 | 10.2 | 1.0x |
| numpy | float32 | 64 | 274,625 | 20.6 | 1.0x |
| cupy | float64 | 64 | 274,625 | 40.6 | 4.0x |
| cupy | float32 | 64 | 274,625 | 150.8 | 7.3x |
| cupy-fused | float64 | 64 | 274,625 | 231.3 | 22.6x |
| cupy-fused | float32 | 64 | 274,625 | 2806.8 | 136.0x |
| numpy | float64 | 128 | 2,146,689 | 9.2 | 1.0x |
| numpy | float32 | 128 | 2,146,689 | 22.3 | 1.0x |
| cupy | float64 | 128 | 2,146,689 | 160.6 | 17.4x |
| cupy | float32 | 128 | 2,146,689 | 413.6 | 18.5x |
| cupy-fused | float64 | 128 | 2,146,689 | 606.9 | 65.9x |
| cupy-fused | float32 | 128 | 2,146,689 | 4026.2 | 180.4x |
| numpy | float64 | 192 | 7,189,057 | 5.3 | 1.0x |
| numpy | float32 | 192 | 7,189,057 | 17.3 | 1.0x |
| cupy | float64 | 192 | 7,189,057 | 183.4 | 34.3x |
| cupy | float32 | 192 | 7,189,057 | 422.2 | 24.4x |
| cupy-fused | float64 | 192 | 7,189,057 | 596.5 | 111.5x |
| cupy-fused | float32 | 192 | 7,189,057 | 4003.7 | 231.2x |
| cupy | float64 | 256 | 16,974,593 | 188.2 | n/a (CPU too slow) |
| cupy | float32 | 256 | 16,974,593 | 425.8 | n/a (CPU too slow) |
| cupy-fused | float64 | 256 | 16,974,593 | 580.6 | n/a (CPU too slow) |
| cupy-fused | float32 | 256 | 16,974,593 | 3864.6 | n/a (CPU too slow) |

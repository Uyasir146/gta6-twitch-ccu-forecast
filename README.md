# gta6-twitch-ccu-forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Uyasir146/gta6-twitch-ccu-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Uyasir146/gta6-twitch-ccu-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

An entry for make.fun's *Forecast GTA VI's launch-day Twitch viewers hour by hour* bounty. Given hour 1's
print, it forecasts the five quantiles (5%, 25%, 50%, 75%, 95%) of Twitch's concurrent viewers in the
"Grand Theft Auto VI" category for hours 2 to 24 of 19 November 2026, scored by a pinball loss.

The forecast is the shape of a launch day, as a multiple of the hour-1 print, taken from 38-42 past big
launches that the case ships as comparables. Each comparable's log-ratio curve `ln(viewers(h)/p1)` is
weighted by a Gaussian kernel on `ln(p1)`, because the shape depends on scale: a launch already enormous in
hour 1 (Elden Ring, Cyberpunk) decays inside a narrow band and recovers for US prime time, while a small
category's hour 1 is a quiet hour and its day runs up several log units later. The central curve is a
weighted local-linear fit in that scale, held flat above the largest comparable, since GTA VI's hour 1 will
very likely be bigger than anything in the set. The band blends the empirical quantiles of the neighbours
with a parametric band on the local residual scale, which keeps the spread honest when only two or three
comparables are close.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

`main.py` is Python 3.13, standard library only, deterministic, and reads every `*.in` case in the input
directory, writing `<id>.out` next to it. Measured leave-one-out over the comparables, scoring each launch
as if its hour 1 were the only known print: pinball loss 0.204 on launches with an hour-1 print of 50k or
more, against 0.251 for an unweighted neighbour forecast.

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
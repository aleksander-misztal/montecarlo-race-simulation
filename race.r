set.seed(42) 

# ----------------------- Data structures -----------------------------
Car <- function(name, base_speed, speed_sd, crash_prob,
                pit_mean, pit_sd, pit_laps = integer(0),
                tyre_decay = 0, boost_laps = integer(0),
                boost_speed = 0, boost_crash = 0) {
  list(name = name,
       base_speed = base_speed,
       speed_sd = speed_sd,
       crash_prob = crash_prob,
       pit_mean = pit_mean,
       pit_sd = pit_sd,
       pit_laps = pit_laps,
       tyre_decay = tyre_decay,
       boost_laps = boost_laps,
       boost_speed = boost_speed,
       boost_crash = boost_crash)
}

Track <- function(lap_length, n_laps, weather_variance = 0) {
  list(lap_length = lap_length,
       n_laps = n_laps,
       weather_variance = weather_variance)
}

# ------------------------ Low‑level helpers --------------------------
lap_time <- function(car, lap, last_pit_lap, track) {
  laps_since_pit <- lap - last_pit_lap - 1  # 0 on first lap after stop
  tyre_penalty   <- car$base_speed * car$tyre_decay * laps_since_pit

  speed <- rnorm(1, car$base_speed, car$speed_sd) -
           tyre_penalty +
           if (lap %in% car$boost_laps) car$boost_speed else 0 +
           rnorm(1, 0, track$weather_variance)
  speed <- max(speed, 1e-3)
  track$lap_length / speed  # seconds
}

pit_stop_time <- function(car) {
  max(rnorm(1, car$pit_mean, car$pit_sd), 0)
}

# -------------------- One‑race simulation ----------------------------
simulate_once <- function(cars, track) {
  results <- vector("list", length(cars))
  names(results) <- vapply(cars, `[[`, "name", FUN.VALUE = character(1))

  for (ci in seq_along(cars)) {
    car <- cars[[ci]]
    total_time <- 0
    status <- "Finished"
    last_pit <- 0

    for (lap in seq_len(track$n_laps)) {
      crash_prob <- car$crash_prob + if (lap %in% car$boost_laps) car$boost_crash else 0
      if (runif(1) < crash_prob) {  # crash!
        status <- sprintf("DNF on lap %d", lap)
        break
      }

      total_time <- total_time + lap_time(car, lap, last_pit, track)

      if (lap %in% car$pit_laps) {
        total_time <- total_time + pit_stop_time(car)
        last_pit <- lap
      }
    }

    results[[ci]] <- list(time  = if (status == "Finished") total_time else NA_real_,
                          status = status,
                          laps   = if (status == "Finished") track$n_laps else lap)
  }
  results
}

# ----------------- Many‑race Monte Carlo wrapper ---------------------
monte_carlo <- function(cars, track, n_iter = 10000, progress = FALSE) {
  car_names <- vapply(cars, `[[`, "name", FUN.VALUE = character(1))
  win <- podium <- dnfs <- setNames(numeric(length(cars)), car_names)
  times <- setNames(vector("list", length(cars)), car_names)

  for (iter in seq_len(n_iter)) {
    res <- simulate_once(cars, track)
    finished <- Filter(function(x) x$status == "Finished", res)

    if (length(finished) > 0) {
      order <- names(sort(vapply(finished, `[[`, "time", FUN.VALUE = numeric(1))))
      win[order[1]] <- win[order[1]] + 1
      for (nm in order[seq_len(min(3, length(order)))]) podium[nm] <- podium[nm] + 1
    }

    for (nm in names(res)) {
      if (res[[nm]]$status == "Finished") {
        times[[nm]] <- c(times[[nm]], res[[nm]]$time)
      } else {
        dnfs[nm] <- dnfs[nm] + 1
      }
    }
    if (progress && iter %% progress == 0) cat("Simulated", iter, "races\n")
  }

  data.frame(Car       = car_names,
             WinPct    = 100 * win / n_iter,
             PodiumPct = 100 * podium / n_iter,
             DNFPct    = 100 * dnfs / n_iter,
             AvgTime   = vapply(times, function(t) if (length(t)) mean(t) else NA_real_, numeric(1)),
             check.names = FALSE)[order(-win), ]
}

# ----------------------- Example setup -------------------------------
cars <- list(
  Car("Rocket",   85, 2.5, 0.01, 18, 2,   pit_laps = c(20),        tyre_decay = 0.003, boost_laps = c(5,6),  boost_speed = 4, boost_crash = 0.01),
  Car("Steady",   80, 1.0, 0.002,20, 1.5, pit_laps = c(22),        tyre_decay = 0.002),
  Car("Risky",    87, 4.0, 0.03, 17, 3,   pit_laps = c(18,34),    tyre_decay = 0.004, boost_laps = c(18,34), boost_speed = 5, boost_crash = 0.02),
  Car("Eco",      78, 1.5, 0.001,0,  0,   pit_laps = integer(0),   tyre_decay = 0.001),
  Car("Sprinter", 90, 3.0, 0.015,21, 2,   pit_laps = c(12,28),    tyre_decay = 0.005, boost_laps = c(1,2,30), boost_speed = 6, boost_crash = 0.015),
  Car("Conserve", 76, 1.0, 0.0008,0, 0,   pit_laps = integer(0),   tyre_decay = 0.0005),
  Car("Balanced", 82, 1.8, 0.005,19, 1.8, pit_laps = c(25),       tyre_decay = 0.002, boost_laps = c(35),   boost_speed = 3, boost_crash = 0.005)
)

track <- Track(lap_length = 5000, n_laps = 40, weather_variance = 0.5)

# --------------------- Run Monte Carlo -------------------------------
summary <- monte_carlo(cars, track, n_iter = 8000, progress = TRUE)

print(summary)
write.csv(summary, "race_montecarlo_summary_R.csv", row.names = FALSE)
cat("Saved: race_montecarlo_summary_R.csv\n")

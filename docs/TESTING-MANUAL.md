# Manual test pass — `test/port-fix`

What automated tests cannot check: a real phone, a real camera, a projector,
and a Codespace whose ports move. 362 tests pass; none of them has an eye.

Work top to bottom. Each item says what to do and what counts as a pass.

---

## A. The server and the port (the bug that started this)

1. **Clean start.** `tools/devserver.sh start` → prints `up on port 8000 (pid N)`
   and the control token. **Pass:** exactly one PID, `/start` loads.

2. **The stale-copy case — the actual bug.** With the server running,
   `rm .devserver-8000.pid`, then `tools/devserver.sh start` again.
   **Pass:** it prints *"an older copy of this app is on port 8000; replacing
   it"* and ends with exactly one server.
   **Fail (the old behaviour):** two processes, the new one dead, the old one
   still answering with whatever code it booted with.
   Confirm with `tools/devserver.sh status` and
   `lsof -ti tcp:8000 -sTCP:LISTEN | wc -l` → must be `1`.

3. **Somebody else's port.** Occupy 8000 with anything that is not this app,
   then `tools/devserver.sh start`. **Pass:** it refuses to kill the stranger,
   moves to 8001, and tells you to set 8001 to Public in the Ports panel.

4. **The join URL follows the port.** After (3), open `/start`.
   **Pass:** the join address and the QR codes point at **8001**, not 8000.
   This is the half that made the QR scan and the phone get nothing.

5. **Codespaces end to end.** Set the port Public. Scan the join QR **from a
   phone on mobile data, not the office wifi**. **Pass:** the capture page
   loads. **Fail:** anything that works on wifi and not on 4G means the URL
   is still the private forwarded one.

6. **`status` and `stop`.** `tools/devserver.sh status` describes the running
   server; `stop` leaves the port free (`status` says nothing is listening).

---

## B. The participant journey — tap arm

Use the **Tap** QR on `/start`, which issues a fresh device id each scan.

7. **A fresh id each time.** Scan twice from the same phone. **Pass:** two
   participants in `/control`, not one.

8. **Tap and submit.** Five taps, Submit. **Pass:** "Thank you — your 5 marks
   are in", then *"That's everything. You can put the phone down."*

9. **No route into the camera.** On that thank-you screen there must be **no
   button, no offer, nothing about "the other half"**. This is the
   contamination check — a tapper who also gets eye-tracked breaks the
   comparison, and it used to be one tap away.

10. **Nothing tells you to look up.** Check every screen in this arm. The
    phrase "look up at the screen" should appear nowhere — the room is not
    always looking at the projector when the phone finishes.

---

## C. The participant journey — eye-tracked arm

Use the **Track** QR. **This is the arm with the new copy; read it as a
participant would, not as someone who knows what it means.**

11. **The intro screen.** **Pass:** it leads with checking your eyes are
    visible in the preview, then says to get comfortable and *stay still* —
    phone and head both — and that dots will appear one at a time to look
    straight at. **Fail:** anything about tapping *then* tracking, any
    implication this is the second half of something.

12. **Calibration itself.** Nine points. **Pass:** each dot is reachable and
    the tap registers (tapping anywhere counts). Note anything that makes you
    move your head — that is the accuracy budget.

13. **The grade screen.** **Pass:** the error percentage is stated, and the
    next-stage text says to look at the picture naturally with nothing to aim
    at, and to keep the phone and head where they are. **Fail:** any "don't
    tap" phrasing, which only makes sense by contrast with the other arm.

14. **The viewing window and the finish.** **Pass:** picture appears for the
    configured time, then *"That's everything. You can put the phone down."*
    and no navigation away.

15. **Two phones, two arms, at once.** Run B and C simultaneously on two
    handsets. **Pass:** `/control` shows one assigned to each and neither
    drifts into the other's screens.

### C-bis. On a real iPhone, in Safari — not a simulator

Chromium cannot answer these, and both bugs below shipped past a passing
Chromium suite.

15a. **The rehearsal animates on the intro screen.** Before Start, the small
     panel must show the finger fading in low-left and the ring imploding on
     the dot, looping. **Fail:** a grey box with only a static dot and no
     motion — that is a CSS feature Safari dropped silently.

15b. **The ring implodes while you hold.** During calibration, hold and watch
     the dot. **Pass:** the ring visibly shrinks over the half second before
     the point is taken. **Fail:** the point is taken with no visible change,
     which reads to a participant as the phone ignoring them.

15c. **Rotate mid-calibration.** The run should pause and ask for portrait
     back, not quietly keep recording against a fit measured the other way up.

---

## D. The projected display

16. **Attention grid, no numbers.** Turn on the attention grid.
    **Pass:** shaded quadrants over the picture, **no percentages in the
    cells**, and the picture still readable through the shading.

17. **Readable from the back.** Stand where the audience will. **Pass:** you
    can tell which quadrants drew attention. If you cannot, the shading needs
    more contrast — report that, it is the point of the layer.

18. **Full-screen layers.** Charts and raw JSON each take the whole screen and
    are mutually exclusive. **Pass:** turning one on turns the other off.
    **Note for §2.3 of SCOPE.md:** while one is up, you currently need the
    other device to turn it off. Confirm how much that hurts in practice.

---

## D-bis. The layer bar on the display (new)

18a. **It is invisible at rest.** Load `/display` and leave the mouse alone.
     **Pass:** no controls anywhere. This is the audience's screen and a
     control strip parked on it is in the photograph of the talk.

18b. **It wakes and sleeps.** Move the mouse. The bar appears at the bottom;
     stop moving and it goes after ~3s. Resting the pointer *on* it keeps it
     up — otherwise it vanishes under the hand reaching for the next toggle.

18c. **It works from inside a full-screen layer — the one that matters.**
     Turn on Charts. Now move the mouse and turn Charts off again *without
     touching another device*. **Pass:** it works. **Fail:** the bar is
     behind the charts and unclickable, which is the bug this was built to
     fix and which the first build had.

18d. **Opened without a token.** Open `/display` directly (no `?k=`) in a
     browser that has never opened `/control`. The buttons should say
     "needs token" rather than failing silently. Reaching it from the nav or
     from Controls carries the token automatically.

18e. **Controls is run setup only.** `/control` has no layer toggles left —
     just a pointer to the display.

---

## E-bis. Time and rounds on /charts (new)

**Rounds**

E1. **The picker names rounds usefully.** Each entry shows date, image, task,
    fixation count, viewing time and participant counts. **Fail:** two rounds
    distinguishable only by timestamp — that is the pair most likely to be
    compared by mistake.

E2. **A past round loads.** Pick a closed round. The panels, matrix and
    difference maps all change to that sitting.

E3. **A past round uses its own model run.** Collect a round at 5 fixations,
    start a new round, switch the model to 10, then reopen the first round in
    the picker. **Pass:** it still reports being compared against the 5. A
    round compared against today's model instead of the one it was shown
    beside is a wrong result that looks right.

E4. **Comparison is opt-in.** The "Compare with" section is absent until the
    checkbox is ticked. Untick it and it goes away again. **This must never
    appear on /display unless deliberately turned on there.**

E5. **A closed round stops auto-refreshing** but the live one keeps updating —
    a past round re-fetching every 5s would restart the animation under you.

**Time**

E6. **It plays.** The "When they looked there" section animates through
    First / Middle / Last third, with the pairwise agreement updating under
    it. Pause works; the three labels are clickable.

E7. **The scale is shared.** A bin with fewer points must look *sparser*, not
    *stronger*. If an empty-ish bin glows as hot as a full one, the shared
    colour scale has broken.

E8. **Reduced motion.** Turn on Reduce Motion in the OS. **Pass:** it does not
    auto-play, and the frame buttons still work.

**Off-picture gaze**

E9. After a round with real eye tracking, the reported off-picture fraction
    should be plausible against what people did. If it reads 0% for everyone,
    the samples are being filtered somewhere again.

---

## E. The model run selector

19. **Both runs are listed.** `/control` → Fixations. **Pass:** it is a real
    dropdown showing *"5 fixations · 10 observers"* and *"10 fixations · 20
    observers"*, and switching changes what the display draws.

20. **Only two tasks.** The task selector offers **Free viewing** and **Find a
    car**, and nothing else. The seven untrained probes are gone.

21. **No SYNTHETIC banner anywhere**, at any setting.

---

## F. Rounds (new)

22. **The list.** `/rounds` shows every sitting, newest first, with dates,
    tapper and eye-tracked counts, and the active one marked.

23. **Export.** Export the round you just collected. **Pass:** a file
    downloads named `round-<n>-<image>-<YYYYMMDDTHHMMSS>.json`. Open it:
    your taps and gaze sessions are in there.

24. **Import is additive — the one that matters.** With a *live round in
    progress and data in it*, import that file. **Pass:** the live round is
    untouched and still active; a new closed round appears. **Fail:** any
    change at all to the live round. This is the "restore twenty minutes
    before a talk" case.

25. **Import twice.** Import the same file again. **Pass:** "already here",
    no duplicate round.

26. **Restore.** Make the imported round active. **Pass:** the display shows
    that round's data, and the settings it ran under come back with it —
    check the viewing time specifically.

27. **Whole-machine export.** *Export everything*, then confirm the bundle
    imports on a fresh database. This is the Codespace-deletion insurance.

---

## G. The model script (on the Mac, not in the Codespace)

28. **Preflight.**
    ```
    python tools/run_session.py --repo ../DeepGaze3.5-VL \
        --image data/images/street_capybara_sign.jpg \
        --samples 25 --num-fixations 10 --probe-only
    ```
    **Pass:** it reports `MPS built: True`, `MPS available: True`, and an
    `mps` GFLOP/s figure **several times** the `cpu` one.
    **Fail — and this is the thing to catch:** mps and cpu within spitting
    distance of each other, or a warning that `PYTORCH_ENABLE_MPS_FALLBACK`
    is set. Either means the GPU is not doing the work, which is the "it
    worked but took all night" symptom. Do not start the real run.

29. **The estimate.** The probe prints one sample's time and the extrapolated
    total. Sanity check against the known figure: the existing 10-fixation,
    20-observer run took **6.0 hours**, so ≈18 min/sample. An estimate wildly
    below that means something is wrong, not that it got fast.

30. **Checkpointing.** Start the real run, let three or four samples complete,
    Ctrl-C. Re-run with `--resume`. **Pass:** it says how many are already
    done and continues rather than restarting.

31. **The output lands.** On completion, copy the JSON into `data/model/`,
    press *Reload runs from disk* in `/control`, and confirm the new run
    appears in the Fixations dropdown labelled with 25 observers.

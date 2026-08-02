# Start Here

Welcome! This is a friendly tour of the repo so you can build your first
keychain and know your way around.

## What this repo does

You draw something on paper. This repo turns your drawing into a 3D
keychain that a 3D printer can print for real.

Here's the trip your drawing takes:

1. You draw on paper and take a photo of it.
2. The computer traces the outline of your drawing (this is called
   **tracing**).
3. It turns that outline into a solid 3D shape, with a hole at the top so
   you can put it on a keyring.
4. It checks the shape to make sure a printer can actually print it —
   this is called a **quality check**.
5. You open the finished file in a program called Bambu Studio, get it
   ready to print, and print it out.

Right now this repo can only build one kind of thing: **keychains**. Maybe
someday it'll build more.

## Make your first keychain

### Step 1: Draw something and take a photo

Draw a simple picture on plain paper — a shape with a clear, bold outline
works best (a star, an animal, your initial, whatever you like). Thin,
wispy lines are hard for the computer to trace, so press firmly and use a
dark marker or pencil if you have one.

Then take a photo of it with a phone or tablet camera. A few tips make a
huge difference:

- **Lay the paper flat** on a table, not held up in your hand.
- **Shoot straight-on**, with the camera pointed straight down at the
  paper — not at an angle.
- **Use bright, even light.** Near a window in the daytime is great.
  Avoid harsh shadows falling across the paper (including the shadow of
  your own head or the camera!).
- **Fill the frame** with the paper so the drawing is easy to see, and
  make sure the photo is in focus.

### Step 2: Get the photo onto the computer

Get the photo file (a `.jpg` or `.png`) from your phone onto the computer
you're using — for example by AirDropping it, emailing it to yourself, or
plugging your phone in and copying the file over. Remember where you
saved it; you'll need that file's path in Step 3.

### Step 3: Set up the project (only needed the first time)

Open a terminal in this repo's folder and run:

```
uv sync
```

This downloads everything the pipeline needs to run. You only have to do
it once (or again later if the project's dependencies change).

### Step 4: Build the keychain

Now run the actual command, pointing it at your photo:

```
uv run make3d your-drawing.jpg --type keychain --photo
```

A few notes on that command:

- Swap `your-drawing.jpg` for the real path to your photo.
- `--type keychain` tells it what to build (keychains are the only thing
  it knows how to build right now).
- `--photo` is important: it tells the pipeline "this is a photo of a
  paper drawing, not a clean digital picture." With `--photo` on, it
  first evens out the lighting and shadows in your photo, then treats
  your pencil or marker lines as the drawing and the paper as the
  background. Leave `--photo` off only if you're starting from an already
  clean digital image (like something drawn in a paint program) instead
  of a photo.

The command will print out what it's doing, and — if all goes well — a
`report:` line at the end showing the full path to your new design's
quality report (a file called `report.md`, inside your drawing's own
folder under `designs/`).

### Step 5: See what you got

The command created a new folder under `designs/` (named after your
drawing) containing:

- a `.3mf` file — the 3D model, ready for Bambu Studio.
- a `.stl` file — a simpler 3D model, useful for other printers.
- `report.md` — the quality report, in plain English (see "Words you'll
  see" below).
- a copy of your traced artwork and source photo.

Open `report.md` and check the **Verdict** at the top. If it says
`print-ready`, you're good to print! If it says something else, see the
"When something goes wrong" section below.

### Step 6: Print it

1. Open the `.3mf` file in **Bambu Studio**.
2. If your keychain has two colors, map its two parts (`base` and
   `detail`) to the two filament colors loaded in your AMS Lite.
3. Slice it and send it to the printer.
4. If you're printing on a printer that only has one color loaded, use
   the `.stl` file instead — no color mapping needed.

## What every folder is

- **`src/`** — the actual program (the code) that does the tracing,
  building, and checking.
- **`designs/`** — every keychain anyone has built lives here, one folder
  per design, saved with the repo.
- **`profiles/`** — settings files that tell the slicer about the printer
  (a Bambu Lab A1 Mini) and the filament.
- **`scripts/`** — small helper scripts, like ones used to make example
  images for testing.
- **`tests/`** — automated checks that make sure the program still works
  correctly. You'll run these before sharing a change (see "Making the
  repo better").

## Words you'll see

- **Tracing** — turning your drawing's outline into shapes the computer
  can build with, instead of just a flat picture.
- **Silhouette** — the outer outline of your drawing: the overall shape
  of the keychain.
- **Detail layer** — darker marks *inside* your drawing's outline (like
  an eye, a stripe, or a letter) that get raised up as a second color,
  separate from the silhouette.
- **STL** — one common file format for 3D shapes. Most 3D printers and
  slicers can open it. It only holds one color of shape.
- **3MF** — a newer file format for 3D shapes. This repo uses it because
  it can hold more than one named part (like `base` and `detail`), which
  is how two-color printing works.
- **Slicing** — turning a 3D model into the exact path a printer's
  nozzle will follow, layer by layer. Done in Bambu Studio or, behind the
  scenes, by OrcaSlicer during the quality check.
- **Print-ready** — a design that has passed *every* check, including a
  real test slice. Only a design whose report says `print-ready` should
  ever be called that.
- **Quality report** — the `report.md` (and `report.json`) file every
  design gets, listing every check that ran, whether it passed, and any
  warnings.

## When something goes wrong

The pipeline tries hard to explain problems in plain words instead of
just crashing. Here's what the most common messages mean and what to do.

| What you'll see (roughly) | What it means | What to do |
|---|---|---|
| `No artwork found in ...: every pixel looks like background` | The computer couldn't find your drawing in the photo at all. | Retake the photo with better lighting and more contrast between the drawing and the paper. |
| `Everything in ... was removed as speckle` | Your drawing's marks were treated as tiny dust specks and thrown out. | Draw bigger/bolder marks, or scan/photograph at a higher resolution. |
| `The outline of your image has parts thinner than ... Try a larger --size ...` | Some part of your outline is too thin to print at this size — but printing it bigger would fix it. | The message tells you what size would work: run the command again with `--size` set to that number, or make your lines bolder. |
| `The outline of your image has parts thinner than ... no --size clears them ... Use a bolder image with thicker strokes.` | Some lines are so thin compared to the rest of the drawing that no size will save them. | Making it bigger won't help this time — go over the thin lines with a bolder pen or marker, then retake the photo. |
| `This artwork traces to N disconnected islands` | Your drawing has separate floating pieces that don't touch, so a printed keychain would fall apart. | Connect the pieces in your drawing (e.g. add a line joining them), then retake the photo. |
| `A ... mm hole with ... mm of material all round needs a solid disc ...` (a hole-margin message) | There's no spot on your drawing thick enough to safely hold the keyring hole. | Draw a slightly bigger/thicker area near the top of your shape, or make the whole drawing bigger. |
| `The whole image is one dark shape ... so there is no separate detail layer` | Your drawing is one solid color throughout, so it will print in a single color instead of two. | This is fine if you wanted one color! If you wanted two, add lighter and darker areas to your drawing. |
| `No detail regions were found, so the design will be a single-colour silhouette` | Same idea — no inner details were found, so it's printing single-color. | Add darker marks *inside* the outline if you want a second color there. |
| `--detail ... was requested, but the traced image has no detail layer to work with` | You asked for a two-color design, but there wasn't one to find. | Same fix as above — add clear inner details to your drawing. |
| `OrcaSlicer was not found ...` (report says `passes-level-1`, not `print-ready`) | The final print-check step (the slicer) isn't installed on this computer, so the design can't officially be called print-ready yet. | Ask a grown-up to install OrcaSlicer, or set the `ORCASLICER_EXE` setting to point at it, then run the command again. |
| `No such image: ...` | The file path you typed doesn't point at a real file. | Double check the spelling and location of your photo's path. |

If you see a message that isn't in this table, read it anyway — this
pipeline is written to explain problems in plain English, and it usually
tells you exactly what to try next.

## Making the repo better

Found something to fix, or want to try improving something? Here's the
safe way to do it, with a grown-up (jconkle-nut) reviewing before
anything becomes official:

1. **Make your change.** Edit whatever file needs it.
2. **Run the tests** to make sure you didn't break anything:

   ```
   uv run pytest
   ```

   All the tests should pass (you'll see a summary line at the end
   saying how many passed). If something fails, look at what it says and
   try to fix it before moving on.
3. **Make a branch** for your change, so it's separate from the main
   version:

   ```
   git checkout -b my-change-name
   ```
4. **Save your change:**

   ```
   git add <the files you changed>
   git commit -m "a short description of what you changed"
   ```
5. **Push it up** to GitHub:

   ```
   git push -u origin my-change-name
   ```
6. **Open a pull request** from your `urmom422` account: go to the repo
   on GitHub, and you should see a button offering to open a pull request
   from the branch you just pushed. Click it, describe what you changed,
   and submit it.

`jconkle-nut` will review your pull request and merge it in once it's
ready. That's it — nice work!

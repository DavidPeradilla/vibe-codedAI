# AI in C

A small AI project written from scratch in C, with no frameworks. It starts with a matrix library and ends with a **task sorter** you can train on your own examples.

This file explains how to compile, run and train everything, so you don't need to look anything up elsewhere.

---

## 1. What is in this project

```
ai-in-c/
  README.md              this file
  tasks.txt              training examples for the task sorter (you edit this)
  Makefile               shortcuts (needs "make"; Windows users can skip it)
  download_mnist.ps1     downloads the MNIST digit images (Windows)
  start.bat              double-click to compile, train (first time) and chat (Windows)
  data/                  MNIST files go here (created by the download script)
  src/
    matrix.h, matrix.c   the matrix library everything else uses
    perceptron.c         one neuron learning AND / OR / XOR
    nn_xor.c             a small neural network that solves XOR
    mnist.h, mnist.c     loader for MNIST files
    train_mnist.c        digit recognizer, version 1
    train_mnist_v2.c     digit recognizer, version 2 (better and faster)
    taskai.c             the task sorter
  tests/
    test_matrix.c        tests for the matrix library
```

| Program | What it does | Result we measured |
|---|---|---|
| `test_matrix` | Checks the matrix library | 17 tests pass |
| `perceptron` | One neuron learns logic gates | AND, OR: 4/4. XOR: stuck at 2/4 (expected) |
| `nn_xor` | A network with a hidden layer | XOR: 4/4 |
| `train_mnist` | Recognizes handwritten digits | 96.50% test accuracy |
| `train_v2` | Same, with ReLU and Adam | 97.79% test accuracy |
| `taskai` | Sorts your tasks into categories | about 70% on unseen tasks (see section 6) |

---

## 2. Requirements

You need a C compiler (`gcc`). Check with:

```
gcc --version
```

If Windows says it is not recognized, install one and then **close and reopen VS Code**:

```
winget install -e --id BrechtSanders.WinLibs.POSIX.UCRT
```

You do **not** need `make`. All commands below use `gcc` directly.

---

## 3. Important rules for every command

1. **Open a terminal inside the `ai-in-c` folder.** The prompt must end in `...\ai-in-c>`. If it ends in `...\Ai-In-C>` (one folder up), run `cd ai-in-c` first. Wrong folder is the most common cause of `No such file or directory`.
2. **Save the file before compiling** (`Ctrl+S`). Changes in an unsaved file are not compiled.
3. **Compile again after every edit to a `.c` file.** Editing `tasks.txt` does not need compiling, only retraining.
4. On Windows PowerShell, run programs as `.\name.exe`. On Linux or macOS use `./name` and drop the `.exe`.

---

## 4. Compile and run each program

### Matrix tests

```
gcc -std=c11 -Wall -Wextra -O2 -Isrc tests/test_matrix.c src/matrix.c -o test_matrix.exe -lm
.\test_matrix.exe
```

Expected last line: `17 tests, 0 failed`.

### Perceptron (one neuron)

```
gcc -std=c11 -Wall -Wextra -O2 -Isrc src/perceptron.c -o perceptron.exe -lm
.\perceptron.exe
```

### XOR network

```
gcc -std=c11 -Wall -Wextra -O2 -Isrc src/nn_xor.c src/matrix.c -o nn_xor.exe -lm
.\nn_xor.exe
```

### MNIST digits (needs the data first)

Download the data once:

```
powershell -ExecutionPolicy Bypass -File .\download_mnist.ps1
```

The `data` folder must then contain these four files (no `.gz` ending):

```
train-images-idx3-ubyte
train-labels-idx1-ubyte
t10k-images-idx3-ubyte
t10k-labels-idx1-ubyte
```

Version 1:

```
gcc -std=c11 -Wall -Wextra -O2 -Isrc src/train_mnist.c src/mnist.c src/matrix.c -o train_mnist.exe -lm
.\train_mnist.exe
```

Version 2 (faster, more accurate, saves `model.bin`):

```
gcc -std=c11 -Wall -Wextra -O3 -march=native -fopenmp -Isrc src/train_mnist_v2.c src/mnist.c src/matrix.c -o train_v2.exe -lm
.\train_v2.exe
```

If the compiler complains about `-fopenmp`, delete just that word. It still works, only slower.

---

## 5. The task sorter: compile once

```
gcc -std=c11 -Wall -Wextra -O2 -Isrc src/taskai.c src/matrix.c -o taskai.exe -lm
```

Two commands do everything:

```
.\taskai.exe train      learns from tasks.txt and saves model.tai
.\taskai.exe chat       type a task, get its category
```

Both use `tasks.txt` in the current folder. To use another file: `.\taskai.exe train myfile.txt`.

---

## 6. How to train the task sorter (step by step)

### Step 1: understand the training file

`tasks.txt` has one example per line, in this form:

```
category|task text
```

Examples:

```
finance|pay the electricity bill before the deadline
health|book a dentist appointment
study|finish the math homework for tomorrow
```

Rules:

- The part before `|` is the category. The part after is the task.
- Lines starting with `#` and empty lines are ignored.
- Category names are **case-sensitive**: `work` and `Work` are two different categories. Pick lowercase and stay consistent.
- Category names can be up to 31 characters. Use up to **16 categories**.
- Only English letters and numbers are understood. Accented characters and symbols are treated as spaces.
- A line without `|` is skipped, and the program prints a warning.

### Step 2: choose your categories

The starter file has `work`, `study`, `home`, `shopping`, `health`, `finance` and `social`. You can delete them, rename them or add more. The same program works for any labels, for example `urgent|soon|later` or `school|errands|family`.

### Step 3: write good examples

This matters more than anything else in the project.

- Aim for **at least 30 to 50 examples per category**. The minimum the program accepts is 20 examples in total and 2 categories.
- Keep categories **roughly balanced**. If one category has 200 examples and another has 10, the AI will favor the big one.
- Write tasks **the way you actually write them**, with your own words, not textbook sentences.
- Vary the wording. Use different words for the same kind of task.
- Avoid examples that could fit two categories. If "buy gifts for the party" could be `shopping` or `social`, decide once and be consistent.

### Step 4: train

```
.\taskai.exe train
```

You will see something like:

```
Loaded 280 examples in 7 categories: work study home shopping health finance social
Vocabulary: 578 words. Training on 224, testing on 56.

epoch   1   loss = 1.9416   train accuracy =  44.2%   test accuracy =  12.5%
epoch 100   loss = 0.0460   train accuracy = 100.0%   test accuracy =  67.9%
...
epoch 400   loss = 0.0430   train accuracy = 100.0%   test accuracy =  69.6%

Mistakes on the test examples:
  "request a refund from the bank"
      real: finance   guessed: work

Saved model.tai. Now run:  taskai chat
```

How to read it:

| Item | Meaning |
|---|---|
| **Vocabulary** | How many different words it learned from your examples |
| **Training on / testing on** | 80% of examples are used to learn. 20% are kept hidden to test honestly. |
| **loss** | How wrong it is. Lower is better. It should fall quickly. |
| **train accuracy** | Score on examples it studied. Reaching 100% is normal. |
| **test accuracy** | Score on examples it has **never seen**. This is the number that matters. |
| **Mistakes** | The hidden examples it got wrong. Read these to find weak spots in your data. |

A big gap between train accuracy (100%) and test accuracy (for example 70%) means it is partly memorizing. The fix is **more and more varied examples**, not changing settings.

**Note:** the saved model learns from 80% of your examples. The other 20% are used only for testing. This is why adding more examples helps.

### Step 5: use it

```
.\taskai.exe chat
```

Type a task and press Enter:

```
> pay my water bill
  -> finance     98.9%
     health       0.6%
     social       0.3%
```

- If the top guess is below 50%, it says **Not sure** and shows its best guesses.
- If none of your words are known, it says **I don't know any of those words yet**.
- Type `/help` to see the commands and `/quit` to leave.

Commands inside chat:

| Command | What it does |
|---|---|
| `/teach category\|text` | Adds a correct example to `tasks.txt` |
| `/retrain` | Retrains on `tasks.txt` right now and reloads the model, without leaving chat |
| `/help` | Lists the commands |
| `/quit` | Leaves chat |

### Step 6: teach it its mistakes (the improvement loop)

Everything happens inside one chat session, with no retyping of terminal commands:

1. Type a task. If the answer is wrong, add the correct one:

   ```
   /teach study|revise for my chemistry test
   ```

   You can `/teach` several examples in a row. You can also edit `tasks.txt` in VS Code instead.
2. Type `/retrain`. After a few seconds it prints something like:

   ```
   Retrained on 283 examples (7 categories). Test accuracy: 66.1%
   ```

3. Type the same task again and see if it improved.

Repeat. Every round of 10 to 20 new examples from your real life makes it better. **Retraining replaces the old `model.tai`.** Nothing is lost, because the knowledge is in `tasks.txt`.

Note: the test accuracy shown after `/retrain` can move up or down by a few points, even when the AI is improving. The test set is small (about 20% of your examples), and each retrain picks a slightly different test set as your data grows. Judge progress mostly by whether your own real tasks come out right.

### Step 6b: skip the typing (shortcuts)

- **Windows launcher:** double-click `start.bat` in the `ai-in-c` folder. It compiles, trains if there is no model yet, and opens chat. Use it every day.
- **Terminal history:** in the VS Code terminal, press the **Up arrow** to bring back the previous command. No need to retype the long `gcc` line.
- You only need to compile again when you change `taskai.c`. Changing `tasks.txt` never needs compiling.

### Step 7: make a different AI with the same program

Change the categories in `tasks.txt` and retrain. Ideas:

- **Urgency:** `urgent`, `soon`, `later`
- **Mood of a message:** `positive`, `negative`, `neutral`
- **Where a note belongs:** your own folders or projects

Tip: to keep your old data, copy `tasks.txt` to `tasks_backup.txt` first, then use a new file name:

```
.\taskai.exe train urgency.txt
.\taskai.exe chat urgency.txt
```

Warning: both commands read and write the same `model.tai`. Training a new file replaces the old model, so retrain the old file when you want to switch back.

---

## 7. Settings you can change (advanced, optional)

Open `src/taskai.c`. These lines are near the top. Edit, save, then compile again.

| Setting | Default | Meaning |
|---|---|---|
| `HIDDEN` | 32 | Size of the hidden layer. In my tests, other sizes made no real difference. |
| `EPOCHS` | 400 | Number of training rounds. |
| `LEARN_RATE` | 0.01 | Size of each learning step. Too high gives unstable loss. Too low learns slowly. |
| `WEIGHT_DECAY` | 0.003 | Pulls weights toward zero so it memorizes less. |
| `MAX_VOCAB` | 4000 | Maximum different words. Extra words are ignored. |
| `MAX_SAMPLES` | 5000 | Maximum examples. |
| `MAX_LABELS` | 16 | Maximum categories. |

I tested several of these on 8 different train/test splits and the accuracy stayed near 70% every time. Adding data is what raises it.

---

## 8. Troubleshooting

| Message or problem | Cause and fix |
|---|---|
| `make : The term 'make' is not recognized` | Windows has no `make`. Use the `gcc` commands in this file. |
| `gcc : not recognized` | Install the compiler (section 2), then restart VS Code. |
| `fatal error: src/xxx.c: No such file or directory` | You are in the wrong folder. Run `cd ai-in-c`, or check that the file is saved in `src/`. |
| `undefined reference to WinMain` | A `.c` file is empty or unsaved, so `main` is missing. Paste the code in and press `Ctrl+S`. |
| `.\xxx.exe is not recognized` | Compiling failed, so the `.exe` was never made. Fix the compiler error first. |
| `Cannot open data/train-images-idx3-ubyte` | MNIST files are missing or still end in `.gz`. Run the download script again. |
| `is not a valid MNIST image file` | A file is corrupted or half unzipped. Download it again. |
| `start.bat` closes instantly or shows an error | Run it from inside VS Code's terminal (`.\start.bat`) to read the message. Usually `gcc` is not installed or a file is unsaved. |
| `No usable model.tai found. Run: taskai train` | Run `.\taskai.exe train` first, in the same folder you run `chat` from. |
| `Need at least 20 examples and 2 categories.` | Add more lines to `tasks.txt`. |
| `Skipping line without '\|'` | That line in `tasks.txt` has no `\|` between category and text. |
| Category shows up twice, like `work` and `Work` | Category names are case-sensitive. Make them identical. |
| `Cannot open tasks.txt` | Run the program from the `ai-in-c` folder, or pass the file path. |
| Output `nan` or loss jumps up | Learning rate too high. Lower `LEARN_RATE` and recompile. |
| `-fopenmp` error | Remove just that flag from the command. |

---

## 9. Limits to know about

- The task sorter understands **word overlap**, not meaning. A task with words it has never seen gets weak or no guesses.
- It handles English letters and numbers only.
- Accuracy depends almost entirely on how many good examples you give it. With a few hundred examples, expect around 70% on new tasks, not perfect.
- It is not a chat assistant. It does one job: sorting text into the categories you defined.

---

## 10. Quick glossary

| Word | Plain meaning |
|---|---|
| **Neural network** | A stack of simple math steps whose numbers (weights) are adjusted until the answers are good |
| **Training** | Repeatedly adjusting the weights to reduce mistakes on examples |
| **Loss** | A number measuring how wrong the AI is. Training tries to make it small. |
| **Epoch** | One full pass over all the training examples |
| **Test accuracy** | Score on examples the AI never trained on. The honest measure. |
| **Overfitting** | Memorizing the training examples instead of learning general patterns |
| **Vocabulary** | The list of words the task sorter knows |
| **Backpropagation** | The method that works out how to adjust each weight, going from the output back to the input |

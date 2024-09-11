# Checked-MPVR

To run the experiment first install the requirements with:

```bash
conda create --name <env> --file requirements.txt
```

Then fill the `.env` API keys and experiment settings.

After that, download the dataset you need for the experiment from the Internet and put it into its
folder in `datasets`. After that, execute the script starting with an only underscore inside that folder with:

```bash
python datasets/<nick</_<file_name>.py
```

Finally, run:

```bash
python main.py
```

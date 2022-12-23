import pathlib

cwd = pathlib.Path().resolve()
filedir = pathlib.Path(__file__).parent.resolve()


class Subject:
    def __init__(self, subj_folder: str,  collection_df=None, group=None, sex=None, age=None):
        if collection_df is None and (group is None or sex is None or age is None):
            raise ValueError("ERROR: Subject instatiation requires either a dataframe of collection info or group, sex, and age values!")
        
        subj_location = pathlib.Path(cwd, subj_folder).resolve()

        # The subject name is the parent folder (i.e 005_S_0221)
        self.name = subj_location.name

        # Setting sample sex group and age
        if collection_df is None:
            self.group = group
            self.sex = sex
            self.age = int(age)
        else:
            entry = collection_df[collection_df["Subject"] == self.name]
            self.group = entry["Group"]
            self.sex = entry["Sex"]
            self.age = entry["Age"]


s = Subject("file", group="AD",sex="M",age=20)

print(s.__dict__)
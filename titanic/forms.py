from django import forms


class TitanicForm(forms.Form):

    sex=forms.ChoiceField(choices=[(0, "Female"), (1, "Male") ])
    pclass=forms.ChoiceField(choices=[(1, "First"), (2, "Second"), (3, "Third") ], label="Passenger Class")
    age=forms.IntegerField()
    relatives=forms.IntegerField()
    fare=forms.FloatField()
	
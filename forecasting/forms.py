from django import forms

class NameForm(forms.Form):
    out_steps = forms.IntegerField()
    conv_width = forms.IntegerField()
    max_epochs = forms.IntegerField()
    #source = forms.CharField(       # A hidden input for internal use
    #    max_length=50,              # tell from which page the user sent the message
    #    widget=forms.HiddenInput()
    #)

def clean(self):
	cleaned_data = super(NameForm, self).clean()
	out_steps = cleaned_data.get('out_steps')
	conv_width = cleaned_data.get('conv_width')
	max_epochs = cleaned_data.get('max_epochs')
	if not name and not email and not message:
		raise forms.ValidationError('You have to write something!')

from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.core.validators import MinLengthValidator
from .models import CustomUser, UserProfile


class CustomUserCreationForm(UserCreationForm):
    username = forms.CharField(
        max_length=150,
        validators=[MinLengthValidator(3)],
        help_text="Минимум 3 символа",
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Введите имя пользователя'})
    )
    password1 = forms.CharField(
        widget=forms.PasswordInput(attrs={'class': 'form-control', 'placeholder': 'Введите пароль'}),
        help_text="Минимум 4 символа"
    )
    password2 = forms.CharField(
        widget=forms.PasswordInput(attrs={'class': 'form-control', 'placeholder': 'Повторите пароль'})
    )
    avatar = forms.ImageField(
        required=False,
        widget=forms.FileInput(attrs={'class': 'form-control'}),
        help_text="Необязательно"
    )

    class Meta:
        model = CustomUser
        fields = ('username', 'password1', 'password2', 'avatar')

    def clean_username(self):
        username = self.cleaned_data.get('username')
        if len(username) < 3:
            raise forms.ValidationError("Имя пользователя должно содержать минимум 3 символа")
        return username


class UserProfileForm(forms.ModelForm):
    avatar = forms.ImageField(required=False, widget=forms.FileInput(attrs={'class': 'form-control'}))

    class Meta:
        model = CustomUser
        fields = ('avatar', 'bio', 'email')
        widgets = {
            'bio': forms.Textarea(attrs={'class': 'form-control', 'rows': 4, 'placeholder': 'Расскажите о себе...'}),
            'email': forms.EmailInput(attrs={'class': 'form-control', 'placeholder': 'Введите email'})
        }

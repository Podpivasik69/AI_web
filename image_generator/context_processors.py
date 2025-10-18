from .models import UserProfile


def user_profile(request):
    """Добавляет профиль пользователя в контекст"""
    if request.user.is_authenticated:
        try:
            # Используем get_or_create для профиля
            profile, created = UserProfile.objects.get_or_create(user=request.user)
            return {'user_profile': profile}
        except Exception as e:
            print(f"Error in context processor: {e}")
            return {}
    return {}

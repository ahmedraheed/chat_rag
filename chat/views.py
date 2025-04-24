# chat/views.py
from django.http import JsonResponse
from .rag_utils import query_rag


def chat_with_bot(request):
    prompt = request.GET.get("prompt", "")
    if not prompt:
        return JsonResponse({"error": "No prompt provided"}, status=400)

    response = query_rag(prompt)
    return JsonResponse({"response": response})

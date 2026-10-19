using System;
using UnityEngine;
using UnityEngine.EventSystems;

namespace Hindsight.UI
{
    /// <summary>Forwards click and hover on a UI rect to callbacks. Used by PlayNavigator.</summary>
    public class NavCell : MonoBehaviour, IPointerClickHandler, IPointerEnterHandler, IPointerExitHandler
    {
        public Action Clicked;
        public Action<bool> Hovered;

        public void OnPointerClick(PointerEventData eventData)
        {
            if (eventData.button == PointerEventData.InputButton.Left) Clicked?.Invoke();
        }

        public void OnPointerEnter(PointerEventData eventData) => Hovered?.Invoke(true);
        public void OnPointerExit(PointerEventData eventData) => Hovered?.Invoke(false);
    }
}

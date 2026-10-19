using System;
using System.Collections.Generic;
using Hindsight.Data;
using Hindsight.Playback;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

namespace Hindsight.UI
{
    /// <summary>
    /// Fixed-width accordion bar for picking a play: quarters > drives > plays.
    /// Exactly one quarter and one drive are open at a time, always the ones containing the
    /// current play, so the bar follows keyboard navigation too. Opening a node takes width
    /// from its collapsed siblings; the bar's total width never changes. Clicking a quarter or
    /// drive jumps to its first play.
    /// </summary>
    public class PlayNavigator : MonoBehaviour
    {
        const float Gap = 3f;
        const float Pad = 4f;
        const float QuarterCollapsedWidth = 84f;
        const float DriveCollapsedWidth = 58f;
        const float QuarterHeaderHeight = 22f;
        const float DriveHeaderHeight = 20f;
        const float AnimationRate = 14f;

        static readonly Color BarColor = new Color32(0x10, 0x16, 0x1D, 0xEB);
        static readonly Color QuarterColor = new Color32(0x1E, 0x27, 0x31, 0xFF);
        static readonly Color QuarterOpenColor = new Color32(0x26, 0x32, 0x41, 0xFF);
        static readonly Color DriveColor = new Color32(0x2B, 0x36, 0x44, 0xFF);
        static readonly Color DriveOpenColor = new Color32(0x34, 0x46, 0x5A, 0xFF);
        static readonly Color MutedText = new Color32(0xAA, 0xB4, 0xC0, 0xFF);

        enum Kind { Quarter, Drive, Play }

        class Node
        {
            public Kind Kind;
            public RectTransform Rect;
            public Image Background;
            public TextMeshProUGUI Label;
            public Outline Outline;          // plays only: selection frame
            public RectTransform Content;    // quarters/drives: holds children
            public readonly List<Node> Children = new();
            public int First, Last;          // play index range covered
            public bool Selected;
            public float Weight;             // animated 0..1 share of the expandable width
            public string CollapsedText, ExpandedText, HoverText;
            public Color BaseColor;

            public bool Contains(int index) => index >= First && index <= Last;
        }

        PlayLoader loader;
        Action<string> hoverChanged;
        RectTransform bar;
        readonly List<Node> quarters = new();
        int shownIndex = -1;
        Node hovered;

        public void Init(PlayLoader playLoader, Action<string> onHoverChanged)
        {
            loader = playLoader;
            hoverChanged = onHoverChanged;
            bar = (RectTransform)transform;
            gameObject.AddComponent<Image>().color = BarColor;
        }

        void LateUpdate()
        {
            if (loader == null) return;
            if (quarters.Count == 0 && loader.Plays.Count > 0) Build();
            if (quarters.Count == 0) return;

            if (loader.CurrentIndex != shownIndex) Select(loader.CurrentIndex);

            float k = 1f - Mathf.Exp(-AnimationRate * Time.unscaledDeltaTime);
            Rect r = bar.rect;
            LayoutAccordion(quarters, Pad, Pad, r.width - 2f * Pad, r.height - 2f * Pad, QuarterCollapsedWidth, k);
        }

        // ---------- Layout ----------

        void LayoutAccordion(List<Node> nodes, float x, float y, float width, float height, float collapsedWidth, float k)
        {
            int n = nodes.Count;
            float gaps = Gap * (n - 1);
            // Collapsed siblings never take more than half the row.
            float collapsed = n > 1 ? Mathf.Min(collapsedWidth, (width - gaps) * 0.5f / (n - 1)) : 0f;
            float extra = Mathf.Max(0f, width - gaps - collapsed * n);

            float sum = 0f;
            foreach (Node node in nodes)
            {
                node.Weight = Mathf.Lerp(node.Weight, node.Selected ? 1f : 0f, k);
                sum += node.Weight;
            }

            foreach (Node node in nodes)
            {
                float share = sum > 1e-4f ? node.Weight / sum : 1f / n;
                float w = collapsed + extra * share;
                Place(node.Rect, x, y, w, height);
                x += w + Gap;

                if (!node.Selected) continue;
                if (node.Kind == Kind.Quarter)
                {
                    LayoutAccordion(node.Children, 0f, 0f, w - 2f * Pad, height - QuarterHeaderHeight - Pad,
                        DriveCollapsedWidth, k);
                }
                else if (node.Kind == Kind.Drive)
                {
                    LayoutEven(node.Children, w - 2f * Pad, height - DriveHeaderHeight - Pad);
                }
            }
        }

        static void LayoutEven(List<Node> nodes, float width, float height)
        {
            int n = nodes.Count;
            float w = (width - Gap * (n - 1)) / n;
            for (int i = 0; i < n; i++)
                Place(nodes[i].Rect, i * (w + Gap), 0f, w, height);
        }

        static void Place(RectTransform rt, float x, float y, float width, float height)
        {
            rt.anchoredPosition = new Vector2(x, -y);
            rt.sizeDelta = new Vector2(Mathf.Max(0f, width), Mathf.Max(0f, height));
        }

        // ---------- Selection ----------

        void Select(int index)
        {
            shownIndex = index;
            foreach (Node q in quarters)
            {
                bool opening = q.Contains(index) && !q.Selected;
                SetSelected(q, q.Contains(index));
                foreach (Node d in q.Children)
                {
                    SetSelected(d, d.Contains(index));
                    // A quarter that just opened shows its drives at their final widths.
                    if (opening) d.Weight = d.Selected ? 1f : 0f;
                    foreach (Node p in d.Children) SetSelected(p, p.Contains(index));
                }
            }
        }

        static void SetSelected(Node node, bool selected)
        {
            node.Selected = selected;
            switch (node.Kind)
            {
                case Kind.Play:
                    node.Outline.enabled = selected;
                    node.Background.color = selected ? Color.Lerp(node.BaseColor, Color.white, 0.25f) : node.BaseColor;
                    node.Label.fontStyle = selected ? FontStyles.Bold : FontStyles.Normal;
                    break;
                default:
                    node.Content.gameObject.SetActive(selected);
                    node.Background.color = selected
                        ? (node.Kind == Kind.Quarter ? QuarterOpenColor : DriveOpenColor)
                        : node.BaseColor;
                    float header = node.Kind == Kind.Quarter ? QuarterHeaderHeight : DriveHeaderHeight;
                    var label = node.Label.rectTransform;
                    // Open: label is a header strip at the top. Closed: it fills the box.
                    label.anchorMin = new Vector2(0f, selected ? 1f : 0f);
                    label.anchorMax = Vector2.one;
                    label.offsetMin = new Vector2(6f, selected ? -header : 0f);
                    label.offsetMax = new Vector2(-4f, 0f);
                    node.Label.alignment = selected ? TextAlignmentOptions.MidlineLeft : TextAlignmentOptions.Center;
                    node.Label.text = selected ? node.ExpandedText : node.CollapsedText;
                    break;
            }
        }

        // ---------- Build ----------

        void Build()
        {
            IReadOnlyList<PlayIndexEntry> plays = loader.Plays;
            Node quarter = null, drive = null;

            for (int i = 0; i < plays.Count; i++)
            {
                PlayIndexEntry e = plays[i];
                if (quarter == null || e.quarter != plays[i - 1].quarter)
                {
                    quarter = CreateBox(Kind.Quarter, bar, QuarterColor, QuarterHeaderHeight, i);
                    quarters.Add(quarter);
                    drive = null;
                }
                if (drive == null || e.driveNumber != plays[i - 1].driveNumber)
                {
                    drive = CreateBox(Kind.Drive, quarter.Content, DriveColor, DriveHeaderHeight, i);
                    AddTeamStripe(drive, TeamColors.For(e.possessionTeam, true).Jersey);
                    quarter.Children.Add(drive);
                }
                drive.Children.Add(CreatePlay(drive.Content, e, i));
                quarter.Last = drive.Last = i;
            }

            foreach (Node q in quarters)
            {
                string name = QuarterName(plays[q.First].quarter);
                q.CollapsedText = $"{name}\n<size=70%><color=#{Hex(MutedText)}>{q.Children.Count} drives</color></size>";
                q.ExpandedText = $"<b>{name}</b>";
                q.HoverText = $"{name}: {q.Children.Count} drives, {q.Last - q.First + 1} plays";
                foreach (Node d in q.Children)
                {
                    PlayIndexEntry first = plays[d.First], last = plays[d.Last];
                    int count = d.Last - d.First + 1;
                    d.CollapsedText = $"D{first.driveNumber}\n<size=75%>{first.possessionTeam}</size>";
                    d.ExpandedText = $"<b>Drive {first.driveNumber}</b>  {first.possessionTeam}  " +
                                     $"<color=#{Hex(MutedText)}>{count} plays</color>";
                    d.HoverText = $"Drive {first.driveNumber} ({first.possessionTeam}): {count} plays, " +
                                  $"{name} {first.gameClock} to {last.gameClock}";
                }
            }

            Select(loader.CurrentIndex < 0 ? 0 : loader.CurrentIndex);
            foreach (Node q in quarters)
            {
                q.Weight = q.Selected ? 1f : 0f;
                foreach (Node d in q.Children) d.Weight = d.Selected ? 1f : 0f;
            }
        }

        Node CreateBox(Kind kind, RectTransform parent, Color color, float headerHeight, int firstIndex)
        {
            var node = new Node { Kind = kind, BaseColor = color, First = firstIndex, Last = firstIndex };
            node.Rect = CreateRect(kind.ToString(), parent);
            node.Background = node.Rect.gameObject.AddComponent<Image>();
            node.Background.color = color;

            node.Content = CreateRect("Content", node.Rect);
            node.Content.anchorMin = Vector2.zero;
            node.Content.anchorMax = Vector2.one;
            node.Content.offsetMin = new Vector2(Pad, Pad);
            node.Content.offsetMax = new Vector2(-Pad, -headerHeight);
            node.Content.pivot = new Vector2(0f, 1f);

            node.Label = CreateLabel(node.Rect, 15f);
            node.Label.color = Color.white;

            Hook(node);
            return node;
        }

        Node CreatePlay(RectTransform parent, PlayIndexEntry e, int index)
        {
            var node = new Node { Kind = Kind.Play, First = index, Last = index, BaseColor = PlayColor(e) };
            node.Rect = CreateRect($"Play {e.playId}", parent);
            node.Background = node.Rect.gameObject.AddComponent<Image>();
            node.Background.color = node.BaseColor;
            node.Outline = node.Rect.gameObject.AddComponent<Outline>();
            node.Outline.effectColor = Color.white;
            node.Outline.effectDistance = new Vector2(2f, -2f);
            node.Outline.enabled = false;

            node.Label = CreateLabel(node.Rect, 14f);
            var rt = node.Label.rectTransform;
            rt.anchorMin = Vector2.zero;
            rt.anchorMax = Vector2.one;
            rt.offsetMin = new Vector2(2f, 2f);
            rt.offsetMax = new Vector2(-2f, -2f);
            node.Label.enableAutoSizing = true;
            node.Label.fontSizeMin = 8f;
            node.Label.fontSizeMax = 15f;
            node.Label.color = e.touchdown ? new Color32(0x1A, 0x1A, 0x1A, 0xFF) : Color.white;
            node.Label.text = $"<size=80%>{e.down}&{e.yardsToGo}</size>\n{ResultText(e)}";

            node.HoverText = $"{QuarterName(e.quarter)} {e.gameClock}  {e.possessionTeam} {Ordinal(e.down)} & {e.yardsToGo}  " +
                             $"<color=#{Hex(MutedText)}>{e.description}</color>";
            Hook(node);
            return node;
        }

        void Hook(Node node)
        {
            var cell = node.Rect.gameObject.AddComponent<NavCell>();
            cell.Clicked = () =>
            {
                if (node.Kind != Kind.Play && node.Selected) return; // already open
                _ = loader.LoadAtAsync(node.First);
            };
            cell.Hovered = isOver =>
            {
                if (isOver) hovered = node;
                else if (hovered == node) hovered = null;
                hoverChanged?.Invoke(hovered?.HoverText);
            };
        }

        static void AddTeamStripe(Node drive, Color color)
        {
            var stripe = CreateRect("TeamStripe", drive.Rect);
            stripe.anchorMin = new Vector2(0f, 1f);
            stripe.anchorMax = Vector2.one;
            stripe.offsetMin = new Vector2(0f, -3f);
            stripe.offsetMax = Vector2.zero;
            var image = stripe.gameObject.AddComponent<Image>();
            image.color = color;
            image.raycastTarget = false;
        }

        static RectTransform CreateRect(string objectName, RectTransform parent)
        {
            var go = new GameObject(objectName, typeof(RectTransform));
            var rt = (RectTransform)go.transform;
            rt.SetParent(parent, false);
            rt.anchorMin = rt.anchorMax = new Vector2(0f, 1f);
            rt.pivot = new Vector2(0f, 1f);
            return rt;
        }

        static TextMeshProUGUI CreateLabel(RectTransform parent, float size)
        {
            var rt = CreateRect("Label", parent);
            var text = rt.gameObject.AddComponent<TextMeshProUGUI>();
            text.fontSize = size;
            text.alignment = TextAlignmentOptions.Center;
            text.textWrappingMode = TextWrappingModes.NoWrap;
            text.overflowMode = TextOverflowModes.Ellipsis;
            text.raycastTarget = false;
            return text;
        }

        // ---------- Formatting ----------

        static Color PlayColor(PlayIndexEntry e)
        {
            if (e.touchdown) return new Color32(0xD9, 0xA8, 0x21, 0xFF);
            switch (e.playType)
            {
                case "pass":
                    return e.passResult switch
                    {
                        "C" => new Color32(0x3E, 0x9E, 0x57, 0xFF),
                        "IN" => new Color32(0xC9, 0x43, 0x43, 0xFF),
                        _ => new Color32(0x5D, 0x67, 0x73, 0xFF), // incomplete
                    };
                case "run": return new Color32(0x3B, 0x78, 0xC9, 0xFF);
                case "sack": return new Color32(0xD9, 0x77, 0x2E, 0xFF);
                case "scramble": return new Color32(0x2E, 0x9C, 0x9C, 0xFF);
                default: return new Color32(0x4A, 0x52, 0x5C, 0xFF); // kneel, spike
            }
        }

        static string ResultText(PlayIndexEntry e)
        {
            if (e.touchdown) return "TD";
            if (e.passResult == "IN") return "INT";
            if (e.playType == "pass" && e.passResult == "I") return "Inc";
            return e.yardsGained > 0 ? $"+{e.yardsGained}" : e.yardsGained.ToString();
        }

        static string QuarterName(int quarter) => quarter <= 4 ? $"Q{quarter}" : "OT";
        static string Ordinal(int n) => n switch { 1 => "1st", 2 => "2nd", 3 => "3rd", _ => $"{n}th" };
        static string Hex(Color c) => ColorUtility.ToHtmlStringRGB(c);
    }
}

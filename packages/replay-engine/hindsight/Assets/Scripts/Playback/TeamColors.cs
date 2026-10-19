using System.Collections.Generic;
using UnityEngine;

namespace Hindsight.Playback
{
    /// <summary>
    /// Jersey colors per club. Like a real broadcast, the home team wears its color and the
    /// visitor wears white with team-colored numbers, which keeps similar teams (LA/BUF) apart.
    /// </summary>
    public static class TeamColors
    {
        public readonly struct Kit
        {
            public readonly Color Jersey;
            public readonly Color Number;
            public Kit(Color jersey, Color number) { Jersey = jersey; Number = number; }
        }

        static readonly Color AwayWhite = Hex("F2F2F2");

        // (home jersey, home number)
        static readonly Dictionary<string, (string jersey, string number)> Clubs = new()
        {
            ["ARI"] = ("97233F", "FFFFFF"), ["ATL"] = ("A71930", "FFFFFF"), ["BAL"] = ("241773", "FFFFFF"),
            ["BUF"] = ("00338D", "FFFFFF"), ["CAR"] = ("0085CA", "FFFFFF"), ["CHI"] = ("0B162A", "C83803"),
            ["CIN"] = ("FB4F14", "000000"), ["CLE"] = ("311D00", "FF3C00"), ["DAL"] = ("003594", "FFFFFF"),
            ["DEN"] = ("FB4F14", "002244"), ["DET"] = ("0076B6", "FFFFFF"), ["GB"] = ("203731", "FFB612"),
            ["HOU"] = ("03202F", "A71930"), ["IND"] = ("002C5F", "FFFFFF"), ["JAX"] = ("006778", "FFFFFF"),
            ["KC"] = ("E31837", "FFFFFF"), ["LA"] = ("003594", "FFA300"), ["LAC"] = ("0080C6", "FFC20E"),
            ["LV"] = ("101010", "A5ACAF"), ["MIA"] = ("008E97", "FFFFFF"), ["MIN"] = ("4F2683", "FFC62F"),
            ["NE"] = ("002244", "FFFFFF"), ["NO"] = ("101820", "D3BC8D"), ["NYG"] = ("0B2265", "FFFFFF"),
            ["NYJ"] = ("125740", "FFFFFF"), ["PHI"] = ("004C54", "FFFFFF"), ["PIT"] = ("101820", "FFB612"),
            ["SEA"] = ("002244", "69BE28"), ["SF"] = ("AA0000", "FFFFFF"), ["TB"] = ("D50A0A", "FFFFFF"),
            ["TEN"] = ("4B92DB", "0C2340"), ["WAS"] = ("5A1414", "FFB612"),
        };

        public static Kit For(string club, bool isHome)
        {
            Color primary = Hex("808080"), number = Color.white;
            if (club != null && Clubs.TryGetValue(club, out var c))
            {
                primary = Hex(c.jersey);
                number = Hex(c.number);
            }
            return isHome ? new Kit(primary, number) : new Kit(AwayWhite, primary);
        }

        static Color Hex(string hex) => ColorUtility.TryParseHtmlString("#" + hex, out var c) ? c : Color.magenta;
    }
}

"""
Cross-visit consensus and visual outlier detection using VLM-extracted semantic profiles.
Detects unrelated scenes, wrong locations, different business categories, and mismatched facades.
Accurately preserves multi-modal legitimate states (e.g., closed roll-down shutters vs open counters).
"""
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


@dataclass
class VlmImageEvaluation:
    file_name: str
    suspicion_score: float
    is_flagged: bool
    reason: str
    vlm_profile: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VlmOutletResult:
    outlet_id: str
    total_images: int
    evaluations: List[VlmImageEvaluation]
    consensus_category: Optional[str] = None
    consensus_signboard: Optional[str] = None
    consensus_landmarks: List[str] = field(default_factory=list)


class VlmAnomalyDetector:
    """
    Evaluates visual consistency across an outlet's photo series based on:
    1. Scene validity (identifying unrelated non-commercial scenes like residential rooms, roads, selfies)
    2. Storefront business category consistency (identifying wrong-location visits, e.g. pharmacy vs grocery)
    3. Structural & architectural congruence (tin roof, brick facade, roll-down shutter, counter type)
    4. Signboard & branding alignment (permanent business title and sponsor banners)
    """

    def evaluate_outlet(self, outlet_id: str, profiles: List[Dict[str, Any]]) -> VlmOutletResult:
        n_images = len(profiles)
        if n_images == 0:
            return VlmOutletResult(outlet_id, 0, [])

        # ---------------------------------------------------------
        # 1. Normalize and Extract Profiles
        # ---------------------------------------------------------
        parsed_items = []
        for p in profiles:
            fn = p.get("file_name", "")
            prof = p.get("profile", {})

            scene_type = str(prof.get("scene_type", "")).lower().strip()
            category = str(prof.get("business_category", "")).lower().strip()
            signboard = prof.get("signboard_name")
            if signboard and str(signboard).strip().lower() not in ("none", "null", ""):
                signboard = str(signboard).strip()
            else:
                signboard = None

            sponsors = prof.get("brand_sponsors") or []
            if isinstance(sponsors, str):
                sponsors = [sponsors]
            sponsors = [str(s).lower().strip() for s in sponsors if s]

            arch_features = prof.get("architectural_features") or []
            if isinstance(arch_features, str):
                arch_features = [arch_features]
            arch_features = [str(a).lower().strip() for a in arch_features if a]

            colors = prof.get("primary_colors") or []
            if isinstance(colors, str):
                colors = [colors]
            colors = [str(c).lower().strip() for c in colors if c]

            desc = str(prof.get("scene_description", "")).strip()

            parsed_items.append({
                "file_name": fn,
                "profile": prof,
                "scene_type": scene_type,
                "category": category,
                "signboard": signboard,
                "sponsors": set(sponsors),
                "arch_features": set(arch_features),
                "colors": set(colors),
                "desc": desc,
            })

        # ---------------------------------------------------------
        # 2. Form Consensus Outlet Identity
        # ---------------------------------------------------------
        commercial_items = [
            it for it in parsed_items
            if it["scene_type"] in ("storefront_exterior", "storefront_interior", "closed_storefront")
            and it["category"] not in ("non_commercial", "unknown", "")
        ]

        if not commercial_items:
            commercial_items = parsed_items

        # Consensus Category
        cat_counts = Counter(it["category"] for it in commercial_items if it["category"])
        consensus_category = cat_counts.most_common(1)[0][0] if cat_counts else "grocery_general_store"
        cat_agreement_ratio = (cat_counts[consensus_category] / len(commercial_items)) if commercial_items else 0.0

        # Consensus Signboard Name
        sign_counts = Counter(it["signboard"].lower() for it in commercial_items if it["signboard"])
        consensus_sign = sign_counts.most_common(1)[0][0] if sign_counts else None

        # Consensus Architectural Features (present in >= 25% of commercial images)
        all_arch = [feat for it in commercial_items for feat in it["arch_features"]]
        arch_counts = Counter(all_arch)
        min_arch_count = max(2, int(len(commercial_items) * 0.25))
        consensus_landmarks = [feat for feat, cnt in arch_counts.items() if cnt >= min_arch_count]

        # Consensus Brand Sponsors
        all_sponsors = [sp for it in commercial_items for sp in it["sponsors"]]
        sponsor_counts = Counter(all_sponsors)
        consensus_sponsors = {sp for sp, cnt in sponsor_counts.items() if cnt >= 2}

        # ---------------------------------------------------------
        # 3. Multi-Axis Anomaly Evaluation per Image
        # ---------------------------------------------------------
        evaluations = []

        for it in parsed_items:
            fn = it["file_name"]
            st = it["scene_type"]
            cat = it["category"]
            sgn = it["signboard"]
            sps = it["sponsors"]
            arch = it["arch_features"]
            colors = it["colors"]
            desc = it["desc"]

            suspicion_score = 0.02
            is_flagged = False
            reason = ""

            # Check 1: Unrelated Scene (Highest Priority Flag)
            if st == "unrelated_scene" or cat == "non_commercial":
                suspicion_score = 0.95
                is_flagged = True
                detail = f": {desc}" if desc else ""
                reason = f"Unrelated scene - photo does not depict a retail outlet storefront ({detail})"

            elif st == "unclear_blurry":
                suspicion_score = 0.85
                is_flagged = True
                reason = "Severe quality outlier - photo is completely blurry or unidentifiable"

            # Check 2: Legitimate Multi-Modal State (Closed Shutter of the same shop)
            elif st == "closed_storefront" or "shutter" in cat or "roll_down_shutter" in arch:
                # Check if it shares any architectural, sponsor, or color features with consensus
                shares_anchor = bool(arch & set(consensus_landmarks)) or bool(sps & consensus_sponsors)
                if shares_anchor or len(commercial_items) <= 2:
                    suspicion_score = 0.05
                    is_flagged = False
                    reason = "Legitimate visit - storefront closed with roll-down shutter matching outlet facade"
                else:
                    # Shutter with completely divergent structure and zero shared anchors
                    suspicion_score = 0.72
                    is_flagged = True
                    reason = "Suspicious closed storefront - shutter structure and facade diverge from outlet history"

            # Check 3: Business Category Divergence (Wrong Location / Different Shop)
            elif (
                len(commercial_items) >= 3
                and cat_agreement_ratio >= 0.50
                and cat
                and cat != consensus_category
                and not self._is_compatible_category(cat, consensus_category)
            ):
                suspicion_score = 0.88
                is_flagged = True
                c_clean = consensus_category.replace("_", " ")
                cat_clean = cat.replace("_", " ")
                reason = f"Storefront mismatch - {cat_clean} photo inconsistent with outlet's established {c_clean} history"

            # Check 4: Direct Signboard Identity Mismatch
            elif (
                consensus_sign
                and sgn
                and not self._is_sign_compatible(sgn, consensus_sign)
                and len(arch & set(consensus_landmarks)) == 0
            ):
                suspicion_score = 0.90
                is_flagged = True
                reason = f"Signage mismatch - storefront displays '{sgn}' conflicting with outlet identity '{consensus_sign}'"

            # Check 5: Total Architectural & Visual Isolation
            elif (
                len(commercial_items) >= 4
                and len(consensus_landmarks) >= 2
                and len(arch & set(consensus_landmarks)) == 0
                and not (sps & consensus_sponsors)
            ):
                suspicion_score = 0.75
                is_flagged = True
                reason = "Distinct background and structural facade - divergent architectural features compared to the outlet series"

            # Consistent Normal Visit
            else:
                suspicion_score = 0.03
                is_flagged = False
                reason = ""

            evaluations.append(
                VlmImageEvaluation(
                    file_name=fn,
                    suspicion_score=round(suspicion_score, 4),
                    is_flagged=is_flagged,
                    reason=reason,
                    vlm_profile=it["profile"],
                )
            )

        return VlmOutletResult(
            outlet_id=outlet_id,
            total_images=n_images,
            evaluations=evaluations,
            consensus_category=consensus_category,
            consensus_signboard=consensus_sign,
            consensus_landmarks=consensus_landmarks,
        )

    def _is_compatible_category(self, cat1: str, cat2: str) -> bool:
        """Determines if two categories frequently co-exist in Bangladeshi retail (e.g. telecom + grocery)."""
        if cat1 == cat2:
            return True
        pairs = {
            frozenset(["telecom_recharge", "grocery_general_store"]),
            frozenset(["telecom_recharge", "electronics_hardware"]),
            frozenset(["tea_stall_restaurant", "grocery_general_store"]),
        }
        return frozenset([cat1, cat2]) in pairs

    def _is_sign_compatible(self, sign1: str, sign2: str) -> bool:
        """Determines if two signboard strings refer to the same establishment (case & token overlap)."""
        s1 = set(re.findall(r"\w+", sign1.lower()))
        s2 = set(re.findall(r"\w+", sign2.lower()))
        common_words = {"store", "telecom", "enterprise", "shop", "service", "center", "agent", "point"}
        informative_1 = s1 - common_words
        informative_2 = s2 - common_words
        if informative_1 and informative_2 and (informative_1 & informative_2):
            return True
        return False


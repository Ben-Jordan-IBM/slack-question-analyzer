"""Tests for the shared text primitives — announcement/notice detection."""

from slack_question_analyzer.textutil import looks_like_announcement


WEBINAR_PROMO = (
    ":alert-blue: Webinar Alert : Widget 12.1 is in the spotlight! :alert-blue:\n"
    "This session will spotlight how Widget 12.1 helps teams modernize.\n"
    ":but-why: Why should customers care?\n"
    ":small_blue_diamond: Modern runtime readiness\n"
    ":small_blue_diamond: Stronger security\n"
    ":arrow_right3: When: July 8, 2026 | 9:00 AM EDT\n"
    "Register here -> Widget Webcasts\n"
    "Please help spread the word with your customers! :rocket:"
)

SALES_KIT = (
    ":rocket: New & Improved Widget Sales Kit is Here!\n"
    "We're excited to share the latest, refined version of the Sales Kit.\n"
    "* :bullettrain_side: FASP spotlight - enhanced positioning\n"
    "* :movie_camera: Demo-ready assets - a compelling Elevator Pitch\n"
    "* :open_file_folder: Simplified structure - find assets in a few clicks\n"
    "Explore the new kit here: Sales kit. Happy selling! :briefcase:\n"
    "Got questions? :eyes: Send them our way! :rocket:\n"
    "cc: @jsmith, @akumar"
)

WIN_WIRE = (
    ":announcement: New Use case on Seismic - Acme Insurance :announcement:\n"
    ":bulb: In 2025, Acme Insurance went through a major cloud transformation\n"
    "and signed a deal to subscribe.\n"
    ":link: Learn more about the Acme use case in this document.\n"
    ":clap: Thank you @jdoe for driving this successful engagement!"
)

RESOURCE_NOTICE = (
    "Please use this documentation page Capability and feature parity and "
    "this note example.com/abc to be aware of high-level differences in "
    "product and feature availability between Gen1 and Gen2. For example, "
    "MFT is not yet available on Azure for Gen2 due to technical issues."
)


def test_detects_marketing_announcements():
    assert looks_like_announcement(WEBINAR_PROMO)
    assert looks_like_announcement(SALES_KIT)
    assert looks_like_announcement(WIN_WIRE)


def test_detects_resource_notice_without_question_mark():
    assert looks_like_announcement(RESOURCE_NOTICE)


def test_notice_with_a_question_mark_is_not_a_notice():
    # The please-directive shape only applies when nothing is asked
    assert not looks_like_announcement(
        "Please use the new template going forward. Does anyone know "
        "where the old submissions live?")


def test_help_seeking_veto_beats_structure():
    # A real troubleshooting post can be emoji- and bullet-heavy; any
    # first-person ask phrasing must disqualify announcement treatment
    bug_report = (
        "Hi team :wave:\n"
        "Looking for some guidance on a potential defect:\n"
        "* MAT / MFT 10.15 Fix12 (multi-instance setup)\n"
        "* SMTP test from IS works :white_check_mark:\n"
        "* MFT Send Mail fails :x:\n"
        "Can you confirm if this is the same underlying defect? "
        "Any workaround if upgrade isn't immediately possible? :pray:"
    )
    assert not looks_like_announcement(bug_report)


def test_broadcast_adjacent_question_is_not_an_announcement():
    # Mentions a sales play but is a genuine ask
    assert not looks_like_announcement(
        "i apologies if I may not be up to date - do we very recently "
        "release a MFT sales play? could I seek for help if I can be "
        "pointed to the latest materials?")


def test_plain_questions_are_never_announcements():
    assert not looks_like_announcement(
        "How do I configure retry limits for failed transfers?")
    assert not looks_like_announcement(
        "Is MFT (self-managed software) supported on IBM LinuxONE?")


def test_please_advise_trouble_report_is_not_a_notice():
    assert not looks_like_announcement(
        "Hi team. Customer on tenant fx.example.com facing issue where a "
        "Copy Task to Target System is failing due to an antivirus "
        "scanning error. Please advise")


def test_structural_signals_alone_never_suffice():
    # Emoji + bullets with no marketing/CTA content: keep it extractable
    assert not looks_like_announcement(
        "Status update :chart_with_upwards_trend:\n"
        "* transfers ran clean overnight :white_check_mark:\n"
        "* retry queue is empty :white_check_mark:\n"
        "* next maintenance window is Saturday :calendar:")


def test_rank_replies_gratitude_signal():
    """The reply BEFORE 'thanks, that worked!' is almost certainly the fix;
    the bare thanks itself is a confirmation, not an answer."""
    from slack_question_analyzer.textutil import rank_replies
    replies = [
        'Try restarting?',
        'Set transfer.timeout to 300 in the partner settings and restart the listener.',
        'thanks, that worked!',
    ]
    ranked = rank_replies(replies)
    assert ranked[0].startswith('Set transfer.timeout')
    assert ranked[-1] == 'thanks, that worked!'


def test_short_messages_are_never_announcements():
    assert not looks_like_announcement("Register here!")
    assert not looks_like_announcement("")
    assert not looks_like_announcement(None)

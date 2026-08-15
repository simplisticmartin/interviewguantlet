"""SQLAlchemy models.

Split by bounded context rather than dumped in one module:

* ``identity``  - users, candidates, resumes, resume claims
* ``catalog``   - companies, roles, job descriptions, concepts, question corpus
* ``interview`` - sessions, questions asked, answers, evaluations
* ``learning``  - skill graph, misconceptions, study plans
* ``prompts``   - versioned prompt registry (spec section 42)
"""

from gauntlet.db.base import Base
from gauntlet.db.models.catalog import (
    Company,
    CompanyQuestionOccurrence,
    Concept,
    JobDescription,
    Question,
    QuestionFamily,
    QuestionSubmission,
    Role,
)
from gauntlet.db.models.identity import Candidate, Resume, ResumeClaim, User
from gauntlet.db.models.interview import (
    CandidateAnswer,
    Evaluation,
    InterviewQuestion,
    InterviewSession,
    ReplaySession,
)
from gauntlet.db.models.learning import (
    CandidateSkillState,
    Misconception,
    SkillEvidence,
    StudyPlan,
    StudyPlanItem,
)
from gauntlet.db.models.prompts import PromptVersion, Rubric

__all__ = [
    "Base",
    "Candidate",
    "CandidateAnswer",
    "CandidateSkillState",
    "Company",
    "CompanyQuestionOccurrence",
    "Concept",
    "Evaluation",
    "InterviewQuestion",
    "InterviewSession",
    "JobDescription",
    "Misconception",
    "PromptVersion",
    "Question",
    "QuestionFamily",
    "QuestionSubmission",
    "ReplaySession",
    "Resume",
    "ResumeClaim",
    "Role",
    "Rubric",
    "SkillEvidence",
    "StudyPlan",
    "StudyPlanItem",
    "User",
]

"""Registration and login."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from apps.api.deps import auth_rate_limit, get_db
from apps.api.schemas import LoginRequest, RegisterRequest, TokenResponse
from gauntlet.services.auth import AuthError, authenticate, create_access_token, register_user

router = APIRouter(prefix="/auth", tags=["auth"], dependencies=[Depends(auth_rate_limit)])


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def register(payload: RegisterRequest, session: Session = Depends(get_db)) -> TokenResponse:
    try:
        candidate = register_user(
            session, payload.email, payload.password, payload.display_name
        )
    except AuthError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    token = create_access_token(candidate.user_id, candidate.id)
    return TokenResponse(
        access_token=token.access_token,
        expires_in=token.expires_in,
        candidate_id=candidate.id,
        display_name=candidate.display_name,
    )


@router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest, session: Session = Depends(get_db)) -> TokenResponse:
    try:
        candidate = authenticate(session, payload.email, payload.password)
    except AuthError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    token = create_access_token(candidate.user_id, candidate.id)
    return TokenResponse(
        access_token=token.access_token,
        expires_in=token.expires_in,
        candidate_id=candidate.id,
        display_name=candidate.display_name,
    )

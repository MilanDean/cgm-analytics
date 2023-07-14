import TopNav from '../components/TopNav';
import TeamPage from '../components/TeamPage';

export default function Team() {
    return (
        <main className="flex h-screen flex-col items-center p-5 m-2">
            <TopNav />
            <TeamPage />
        </main>
    );
}